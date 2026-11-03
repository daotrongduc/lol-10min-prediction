#!/usr/bin/env python3
"""Khai phá dữ liệu LoL phút 10. Chạy độc lập từ CSV đến CSV/PNG/model.
Không dùng nhãn cuối trận trong đặc trưng. Mọi lựa chọn chỉ dùng Train/Val.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse, hashlib, itertools, json, logging, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
    confusion_matrix,classification_report,roc_curve,silhouette_score,adjusted_rand_score)
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
SEED = 42
SUFFIXES = ['WardsPlaced','WardsDestroyed','FirstBlood','Kills','Deaths','Assists',
 'EliteMonsters','Dragons','Heralds','TowersDestroyed','TotalGold','AvgLevel',
 'TotalExperience','TotalMinionsKilled','TotalJungleMinionsKilled','GoldDiff',
 'ExperienceDiff','CSPerMin','GoldPerMin']
DESCRIPTIONS = {
 'WardsPlaced':('Số mắt đã đặt','mắt'), 'WardsDestroyed':('Số mắt đã phá','mắt'),
 'FirstBlood':('Đội giành mạng đầu tiên; 1=có, 0=không','nhị phân'),
 'Kills':('Số mạng hạ gục','mạng'),'Deaths':('Số lần bị hạ gục','lần'),
 'Assists':('Số lượt hỗ trợ hạ gục','lượt'),'EliteMonsters':('Số quái lớn: rồng + sứ giả','quái'),
 'Dragons':('Số rồng đã hạ','rồng'),'Heralds':('Số sứ giả đã hạ','sứ giả'),
 'TowersDestroyed':('Số trụ đối phương đã phá','trụ'),'TotalGold':('Tổng vàng của đội','vàng'),
 'AvgLevel':('Cấp độ trung bình thành viên','cấp'),'TotalExperience':('Tổng kinh nghiệm','XP'),
 'TotalMinionsKilled':('Số lính đường đã hạ','lính'),'TotalJungleMinionsKilled':('Số quái rừng đã hạ','quái'),
 'GoldDiff':('Chênh lệch vàng đội này trừ đối phương','vàng'),
 'ExperienceDiff':('Chênh lệch kinh nghiệm đội này trừ đối phương','XP'),
 'CSPerMin':('Lính đường hạ trung bình mỗi phút của đội','lính/phút'),
 'GoldPerMin':('Vàng trung bình mỗi phút của đội','vàng/phút')}
DIFF_PAIRS = {'wardPlacedDiff':'WardsPlaced','wardDestroyedDiff':'WardsDestroyed',
 'killDiff':'Kills','assistDiff':'Assists','dragonDiff':'Dragons','heraldDiff':'Heralds',
 'towerDiff':'TowersDestroyed','avgLevelDiff':'AvgLevel','minionDiff':'TotalMinionsKilled',
 'jungleMinionDiff':'TotalJungleMinionsKilled'}
# Loại 13 biến có quan hệ đại số đã kiểm tra, giữ 25 biến gốc độc lập đại số.
DROP_REDUNDANT = ['redDeaths','blueDeaths','redFirstBlood','redGoldDiff','redExperienceDiff',
 'blueGoldDiff','blueExperienceDiff','blueCSPerMin','redCSPerMin',
 'blueGoldPerMin','redGoldPerMin','blueEliteMonsters','redEliteMonsters']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':180})

def save_csv(obj, path):
    pd.DataFrame(obj).to_csv(path,index=False,encoding='utf-8-sig')

def save_json(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def save_fig(path):
    plt.tight_layout(); plt.savefig(path,bbox_inches='tight'); plt.close()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data(path):
    if not path.is_file(): raise FileNotFoundError(f'Không tìm thấy CSV: {path}')
    d=pd.read_csv(path)
    expected=['gameId','blueWins']+[side+s for side in ('blue','red') for s in SUFFIXES]
    if set(d.columns)!=set(expected): raise ValueError(f'Schema khác dự kiến: {set(d.columns)^set(expected)}')
    if not all(pd.api.types.is_numeric_dtype(d[c]) for c in d): raise ValueError('CSV phải là dữ liệu số')
    return d

def inspect_data(d,r):
    summary={'rows':len(d),'columns':d.shape[1],'missing_cells':int(d.isna().sum().sum()),
     'duplicate_rows':int(d.duplicated().sum()),'unique_gameId':int(d.gameId.nunique()),
     'duplicate_gameId':int(d.gameId.duplicated().sum()),'blue_wins':int((d.blueWins==1).sum()),
     'red_wins':int((d.blueWins==0).sum()),'blue_win_rate':float(d.blueWins.mean())}
    save_csv([summary],r/'dataset_summary.csv')
    dictionary=[]
    for c in d:
        side='Blue' if c.startswith('blue') else 'Red'
        if c=='gameId': desc,unit,when='Định danh trận đấu','ID','định danh'
        elif c=='blueWins': desc,unit,when='Kết quả cuối trận; 1=Blue thắng, 0=Red thắng','nhị phân','cuối trận (target)'
        else:
            suffix=c[4:] if side=='Blue' else c[3:]
            desc,unit=DESCRIPTIONS[suffix];desc=f'{side}: {desc}';when='phút 10'
        dictionary.append({'feature':c,'dtype':str(d[c].dtype),'description':desc,'unit':unit,
          'available_at':when,'missing':int(d[c].isna().sum()),'unique':d[c].nunique(),
          'min':d[c].min(),'max':d[c].max()})
    save_csv(dictionary,r/'data_dictionary.csv')
    desc=d.describe().T.reset_index(names='feature').rename(columns={'50%':'median'})
    save_csv(desc,r/'descriptive_statistics.csv')
    checks=[{'check':'finite_and_complete','passed':bool(np.isfinite(d.to_numpy()).all())},
      {'check':'unique_rows','passed':not d.duplicated().any()},
      {'check':'unique_gameId','passed':not d.gameId.duplicated().any()}]
    for c in ['blueWins','blueFirstBlood','redFirstBlood','blueDragons','redDragons','blueHeralds','redHeralds']:
        checks.append({'check':c+'_binary','passed':set(d[c].unique())<={0,1}})
    for c in d:
        if 'Diff' not in c:
            checks.append({'check':c+'_nonnegative','passed':bool((d[c]>=0).all())})
    save_csv(checks,r/'data_quality_checks.csv')
    if not all(x['passed'] for x in checks): raise ValueError('Dữ liệu không qua kiểm tra; xem data_quality_checks.csv, không tự sửa giá trị.')
    return summary

def audit_redundancy(d,r):
    relations=[('redDeaths','blueKills',d.blueKills),('blueDeaths','redKills',d.redKills),
      ('redFirstBlood','1 - blueFirstBlood',1-d.blueFirstBlood),
      ('redGoldDiff','-blueGoldDiff',-d.blueGoldDiff),
      ('redExperienceDiff','-blueExperienceDiff',-d.blueExperienceDiff),
      ('blueGoldDiff','blueTotalGold - redTotalGold',d.blueTotalGold-d.redTotalGold),
      ('blueExperienceDiff','blueTotalExperience - redTotalExperience',d.blueTotalExperience-d.redTotalExperience)]
    for side in ('blue','red'):
        for lhs,rhs,v in [('CSPerMin','TotalMinionsKilled',10),('GoldPerMin','TotalGold',10)]:
            relations.append((side+lhs,f'{side+rhs} / {v}',d[side+rhs]/v))
        relations.append((side+'EliteMonsters',f'{side}Dragons + {side}Heralds',d[side+'Dragons']+d[side+'Heralds']))
    rows=[]
    for a,b,val in relations:
        err=np.abs(d[a]-val)
        rows.append({'feature_A':a,'feature_B_or_expression':b,'relation':'algebraic equality',
         'exact_equal':bool(np.array_equal(d[a].values,val.values)),
         'equal_at_1e-9':bool(np.allclose(d[a],val,atol=1e-9,rtol=0)),
         'max_abs_error':err.max(),'decision_reduced_raw':'drop '+a,
         'reason':'Giữ thành phần gốc; biến suy ra không cung cấp thông tin mới.'})
    save_csv(rows,r/'redundancy_audit.csv')
    if not all(x['equal_at_1e-9'] for x in rows): raise ValueError('Quan hệ dẫn xuất không đúng; cần xem lại feature set.')
    leak=[]
    for c in d:
        leak.append({'feature':c,'role':'target_only' if c=='blueWins' else 'identifier_excluded' if c=='gameId' else 'minute10_predictor',
         'allowed_as_input':c not in ('gameId','blueWins'),
         'reason':'Nhãn cuối trận' if c=='blueWins' else 'Định danh không mô tả trạng thái' if c=='gameId' else 'Nguồn mô tả snapshot phút 10; kiểm tra CSV không tự chứng minh timestamp'})
    save_csv(leak,r/'leakage_audit.csv')

def feature_engineering(d):
    # Chỉ tham chiếu cột phút 10; hàm hoạt động cả khi CSV không có target/gameId.
    raw=d[[side+s for side in ('blue','red') for s in SUFFIXES if side+s not in DROP_REDUNDANT]].copy()
    diff=pd.DataFrame({name:d['blue'+s]-d['red'+s] for name,s in DIFF_PAIRS.items()},index=d.index)
    diff['blueGoldDiff']=d.blueGoldDiff;diff['blueExperienceDiff']=d.blueExperienceDiff
    diff['blueFirstBlood']=d.blueFirstBlood
    assert not {'blueWins','gameId'} & (set(raw.columns) | set(diff.columns))
    return {'reduced_raw':raw,'difference':diff}

def split_data(d,r):
    train,held=train_test_split(d.index,test_size=.2,random_state=SEED,stratify=d.blueWins)
    val,test=train_test_split(held,test_size=.5,random_state=SEED,stratify=d.loc[held,'blueWins'])
    splits={'Train':train,'Validation':val,'Test':test}
    for a,b in itertools.combinations(splits,2): assert not set(d.loc[splits[a],'gameId']) & set(d.loc[splits[b],'gameId'])
    assignments=pd.DataFrame({'row_index':d.index,'gameId':d.gameId,'split':''})
    rows=[]
    for name,idx in splits.items():
        assignments.loc[idx,'split']=name
        rows.append({'split':name,'n':len(idx),'fraction':len(idx)/len(d),'blue_wins':int(d.loc[idx,'blueWins'].sum()),
         'red_wins':int((d.loc[idx,'blueWins']==0).sum()),'blue_win_rate':d.loc[idx,'blueWins'].mean()})
    save_csv(rows,r/'split_summary.csv');save_csv(assignments,r/'split_assignments.csv')
    return splits

def wilson(w,n):
    if n==0:return np.nan,np.nan
    z=1.96;p=w/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def perform_eda(d,r,f):
    x=feature_engineering(d)['difference']; y=d.blueWins
    out=[]
    for c in x:
        q1,q3=x[c].quantile([.25,.75]);iqr=q3-q1
        out.append({'feature':c,'q1_train':q1,'q3_train':q3,'iqr_outliers':int(((x[c]<q1-1.5*iqr)|(x[c]>q3+1.5*iqr)).sum()),'action':'keep; descriptive flag, not confirmed error'})
    # Kiểm tra outlier từng biến gốc, ngưỡng chỉ học từ Train.
    for c in d.drop(columns=['gameId','blueWins']):
        q1,q3=d[c].quantile([.25,.75]);iqr=q3-q1
        out.append({'feature':c,'q1_train':q1,'q3_train':q3,'iqr_outliers':int(((d[c]<q1-1.5*iqr)|(d[c]>q3+1.5*iqr)).sum()),'action':'keep; descriptive flag, not confirmed error'})
    save_csv(out,r/'outlier_audit_train.csv')
    plt.figure(figsize=(5,3));counts=y.value_counts().sort_index();plt.bar(['Red thắng','Blue thắng'],counts,color=['#cd5960','#3487b7']);plt.ylabel('Số trận (Train)');save_fig(f/'target_distribution.png')
    for c,label,unit,stem in [('blueGoldDiff','Chênh lệch vàng','vàng','gold_diff'),('blueExperienceDiff','Chênh lệch kinh nghiệm','XP','experience_diff')]:
        plt.figure(figsize=(6,3));plt.hist(x[c],bins=40,color='#3687a5');plt.axvline(0,color='black',lw=.8);plt.xlabel(f'{label} Blue − Red ({unit})');plt.ylabel('Số trận (Train)');save_fig(f/(stem+'_distribution.png'))
        plt.figure(figsize=(6,3));plt.boxplot([x.loc[y==0,c],x.loc[y==1,c]],tick_labels=['Red thắng','Blue thắng'],showfliers=False);plt.ylabel(f'{label} ({unit})');save_fig(f/(stem+'_by_winner.png'))
    all_rates=[]
    def rates(feature,groups):
        rows=[]
        for g in sorted(groups.dropna().unique()):
            mask=groups==g;n=int(mask.sum());w=int(y[mask].sum());lo,hi=wilson(w,n)
            rows.append({'feature':feature,'group':str(g),'n':n,'blue_wins':w,'win_rate':w/n,'ci_low':lo,'ci_high':hi})
        all_rates.extend(rows);return pd.DataFrame(rows)
    fb=rates('blueFirstBlood',d.blueFirstBlood)
    plt.figure(figsize=(5.5,3));plt.bar(['Red First Blood','Blue First Blood'],fb.win_rate,color=['#cd5960','#3487b7']);plt.ylim(0,1);plt.ylabel('Tỷ lệ Blue thắng (Train)');save_fig(f/'first_blood_winrate.png')
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,c,label in zip(axes,['dragonDiff','heraldDiff','towerDiff'],['Rồng','Sứ giả','Trụ']):
        a=rates(c,np.sign(x[c]));ax.bar(['Red hơn','Bằng','Blue hơn'],a.win_rate);ax.set_title(label);ax.set_ylim(0,1);ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Tỷ lệ Blue thắng')
    save_fig(f/'objective_winrate.png')
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,c in zip(axes.flat,['killDiff','assistDiff','minionDiff','jungleMinionDiff','avgLevelDiff','wardPlacedDiff']):
        a=rates(c,np.sign(x[c]));ax.bar(a.group,a.win_rate);ax.set_title(c);ax.set_ylim(0,1);ax.set_xlabel('Dấu chênh lệch Blue − Red');ax.set_ylabel('Tỷ lệ Blue thắng')
    save_fig(f/'advantage_winrates.png')
    rates('wardDestroyedDiff',np.sign(x.wardDestroyedDiff))
    for c in ['blueGoldDiff','blueExperienceDiff']:
        q=pd.qcut(x[c],q=5,duplicates='drop');rates(c,q.astype(str))
    save_csv(all_rates,r/'eda_winrates.csv')
    corr=x.assign(blueWins=y).corr();save_csv(corr.reset_index(names='feature'),r/'correlations_train.csv')
    plt.figure(figsize=(9,7));plt.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');plt.colorbar(label='Pearson r');plt.xticks(range(len(corr)),corr.columns,rotation=70,ha='right',fontsize=8);plt.yticks(range(len(corr)),corr.index,fontsize=8);save_fig(f/'correlation_heatmap.png')
    plt.figure(figsize=(6,4));sample=d.sample(min(2000,len(d)),random_state=SEED)
    for v,color,label in [(0,'#cd5960','Red thắng'),(1,'#3487b7','Blue thắng')]:
        s=sample[sample.blueWins==v];plt.scatter(s.blueGoldDiff,s.blueExperienceDiff,s=8,alpha=.3,color=color,label=label)
    plt.xlabel('Blue − Red vàng');plt.ylabel('Blue − Red XP');plt.legend();save_fig(f/'gold_experience_scatter.png')
    return corr

def metrics(y,p):
    pred=(np.asarray(p)>=.5).astype(int)
    return {'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
     'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p)}

def make_model(name,params):
    if name=='Dummy':return DummyClassifier(strategy='most_frequent')
    if name=='Logistic Regression':return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=SEED,**params))
    if name=='Decision Tree':return DecisionTreeClassifier(random_state=SEED,**params)
    if name=='Random Forest':return RandomForestClassifier(n_estimators=250,n_jobs=2,random_state=SEED,**params)
    return HistGradientBoostingClassifier(max_iter=150,early_stopping=False,random_state=SEED,**params)

def train_classification_models(features,d,s,r,f):
    grids={'Dummy':[{}],'Logistic Regression':list(ParameterGrid({'C':[.1,1,10]})),
      'Decision Tree':list(ParameterGrid({'max_depth':[3,5,8,12],'min_samples_leaf':[10,30]})),
      'Random Forest':list(ParameterGrid({'max_depth':[6,12,None],'min_samples_leaf':[5,15]})),
      'Histogram Gradient Boosting':list(ParameterGrid({'learning_rate':[.05,.1],'max_leaf_nodes':[7,15,31],'l2_regularization':[1.0]}))}
    save_json({'primary':'validation_f1','secondary':'validation_roc_auc','tie_break':'candidate_id_ascending',
     'threshold':.5,'grids':grids,'fit':'Train only','refit':'Train + Validation','test_policy':'one selected model, one final evaluation'},r/'experiment_protocol.json')
    records=[];models={};cnt=0
    for fs,x in features.items():
        assert not {'gameId','blueWins'} & set(x)
        for name,grid in grids.items():
            for params in grid:
                cnt+=1;model=make_model(name,params);tic=time.perf_counter()
                model.fit(x.loc[s['Train']],d.loc[s['Train'],'blueWins']);elapsed=time.perf_counter()-tic
                pr=model.predict_proba(x.loc[s['Validation']])[:,1]
                rec={'candidate_id':cnt,'feature_set':fs,'n_features':x.shape[1],'model':name,
                 'parameters':json.dumps(params,sort_keys=True),'fit_seconds':elapsed,
                 **metrics(d.loc[s['Validation'],'blueWins'],pr)}
                records.append(rec);models[cnt]=model
            logging.info('Hoàn thành %s / %s',fs,name)
    search=pd.DataFrame(records).sort_values(['f1','roc_auc','candidate_id'],ascending=[False,False,True])
    save_csv(search,r/'classification_search.csv')
    best_family=search.drop_duplicates(['feature_set','model']);save_csv(best_family,r/'classification_validation.csv')
    winner=search.iloc[0].to_dict();save_json(winner,r/'selected_model.json')
    best=models[int(winner['candidate_id'])];x=features[winner['feature_set']]
    comp=search.drop_duplicates('model').sort_values('f1')
    plt.figure(figsize=(7,3));plt.barh(comp.model,comp.f1);plt.xlim(0,1);plt.xlabel('F1 trên Validation, cấu hình tốt nhất từng họ');save_fig(f/'model_comparison.png')
    # Importance được tính trước refit, Validation chưa tham gia fit.
    pi=permutation_importance(best,x.loc[s['Validation']],d.loc[s['Validation'],'blueWins'],scoring='f1',n_repeats=20,random_state=SEED,n_jobs=2)
    importance=pd.DataFrame({'feature':x.columns,'importance_mean':pi.importances_mean,'importance_std':pi.importances_std}).sort_values('importance_mean',ascending=False)
    save_csv(importance,r/'feature_importance.csv')
    a=importance.head(15).iloc[::-1];plt.figure(figsize=(7,4));plt.barh(a.feature,a.importance_mean,xerr=a.importance_std);plt.xlabel('Mức giảm F1 khi hoán vị (Validation; 20 lần)');save_fig(f/'feature_importance.png')
    if hasattr(best,'feature_importances_'):save_csv({'feature':x.columns,'impurity_importance':best.feature_importances_},r/'impurity_importance.csv')
    return winner,best,grids

def perform_ablation(diff,d,s,r,f):
    # So sánh công bằng: cố định LR C=1 cho toàn bộ nhóm, không dùng chọn model cuối.
    groups={'Full difference':list(diff),'Economy only':['blueGoldDiff'],
     'Combat only':['killDiff','assistDiff','blueFirstBlood'],
     'Farming + experience':['minionDiff','jungleMinionDiff','avgLevelDiff','blueExperienceDiff'],
     'Objectives only':['dragonDiff','heraldDiff','towerDiff'],
     'Vision only':['wardPlacedDiff','wardDestroyedDiff'],
     'Without gold':[c for c in diff if c!='blueGoldDiff'],
     'Without gold and XP':[c for c in diff if c not in ['blueGoldDiff','blueExperienceDiff']]}
    rows=[]
    for name,cols in groups.items():
        model=make_model('Logistic Regression',{'C':1});model.fit(diff.loc[s['Train'],cols],d.loc[s['Train'],'blueWins'])
        rows.append({'ablation':name,'features':'|'.join(cols),'n_features':len(cols),**metrics(d.loc[s['Validation'],'blueWins'],model.predict_proba(diff.loc[s['Validation'],cols])[:,1])})
    save_csv(rows,r/'validation_ablation.csv');a=pd.DataFrame(rows).iloc[::-1]
    plt.figure(figsize=(7,4));plt.barh(a.ablation,a.f1);plt.xlim(0,1);plt.xlabel('F1 Validation — LR cố định C=1');save_fig(f/'ablation.png')

def evaluate_test(winner,best,features,d,s,r,f):
    x=features[winner['feature_set']];fit_idx=np.concatenate([s['Train'],s['Validation']]);final=clone(best)
    final.fit(x.loc[fit_idx],d.loc[fit_idx,'blueWins'])
    # Đây là vị trí duy nhất đưa feature Test vào bộ phân loại đã khóa.
    y=d.loc[s['Test'],'blueWins'];p=final.predict_proba(x.loc[s['Test']])[:,1];pred=(p>=.5).astype(int)
    scores=metrics(y,p);save_csv([{'model':winner['model'],'feature_set':winner['feature_set'],**scores}],r/'classification_test.csv')
    save_csv({'gameId':d.loc[s['Test'],'gameId'],'y_true':y,'prob_blue_win':p,'y_pred':pred},r/'test_predictions.csv')
    cm=confusion_matrix(y,pred,labels=[0,1]);save_csv({'actual':['Red Win (0)','Blue Win (1)'],'pred_red':cm[:,0],'pred_blue':cm[:,1]},r/'confusion_matrix.csv')
    save_json(classification_report(y,pred,output_dict=True,zero_division=0),r/'classification_report.json')
    plt.figure(figsize=(4.5,3.5));plt.imshow(cm,cmap='Blues');plt.xticks([0,1],['Red','Blue']);plt.yticks([0,1],['Red','Blue']);plt.xlabel('Dự đoán');plt.ylabel('Thực tế')
    for i in range(2):
        for j in range(2):plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,color='white' if cm[i,j]>cm.max()/2 else 'black')
    save_fig(f/'confusion_matrix.png')
    fp,tp,_=roc_curve(y,p);plt.figure(figsize=(5,3.5));plt.plot(fp,tp,label=f'AUC = {scores["roc_auc"]:.4f}');plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend();save_fig(f/'roc_curve.png')
    lo,hi=wilson(int((y==pred).sum()),len(y))
    meta={'model_family':winner['model'],'feature_set':winner['feature_set'],'feature_names':list(x),
      'parameters':final.get_params(),'positive_class':1,'positive_meaning':'Blue thắng cuối trận',
      'threshold':.5,'seed':SEED,'test_metrics':scores,'accuracy_wilson95':[lo,hi],
      'train_plus_validation_size':len(fit_idx),'test_size':len(y)}
    save_json(meta,r/'final_model_metadata.json')
    joblib.dump({'model':final,'feature_set':winner['feature_set'],'feature_names':list(x),'threshold':.5},r/'final_model.joblib')
    return scores

def clustering_features(d):
    return pd.DataFrame({'absGoldDiff':d.blueGoldDiff.abs(),'absExperienceDiff':d.blueExperienceDiff.abs(),
     'absKillDiff':(d.blueKills-d.redKills).abs(),
     'absMinionDiff':(d.blueTotalMinionsKilled-d.redTotalMinionsKilled).abs(),
     'totalKills':d.blueKills+d.redKills,'totalWardsPlaced':d.blueWardsPlaced+d.redWardsPlaced,
     'totalWardsDestroyed':d.blueWardsDestroyed+d.redWardsDestroyed,
     'totalEliteMonsters':d.blueEliteMonsters+d.redEliteMonsters,
     'totalTowers':d.blueTowersDestroyed+d.redTowersDestroyed},index=d.index)

def perform_clustering(d,r,f):
    c=clustering_features(d);scaler=StandardScaler();z=scaler.fit_transform(c)
    sample=np.random.RandomState(SEED).choice(len(z),min(3000,len(z)),replace=False)
    rows=[];fits={}
    for k in range(2,11):
        model=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(z);fits[k]=model
        score=silhouette_score(z[sample],model.labels_[sample])
        rows.append({'k':k,'inertia':model.inertia_,'silhouette_sample':score,'sample_n':len(sample)})
    ks=pd.DataFrame(rows);save_csv(ks,r/'cluster_k_selection.csv')
    k=int(ks.sort_values(['silhouette_sample','k'],ascending=[False,True]).iloc[0].k);model=fits[k];lab=model.labels_
    for col,stem,label in [('inertia','elbow_curve','Inertia (Train đã chuẩn hóa)'),('silhouette_sample','silhouette_scores','Silhouette (mẫu Train, n=3.000)')]:
        plt.figure(figsize=(6,3));plt.plot(ks.k,ks[col],'-o');plt.axvline(k,color='#ba4c54',ls='--');plt.xlabel('K');plt.ylabel(label);save_fig(f/(stem+'.png'))
    cent=pd.DataFrame(scaler.inverse_transform(model.cluster_centers_),columns=c.columns);cent.insert(0,'cluster',range(k));save_csv(cent,r/'cluster_centroids.csv')
    summaries=[]
    for j in range(k):
        mask=lab==j;n=int(mask.sum());wins=int(d.loc[mask,'blueWins'].sum());lo,hi=wilson(wins,n)
        summaries.append({'cluster':j,'n':n,'fraction':n/len(d),'blue_win_rate':wins/n,'ci_low':lo,'ci_high':hi,**c.loc[mask].mean().to_dict()})
    save_csv(summaries,r/'cluster_summary.csv')
    save_csv({'gameId':d.gameId,'cluster':lab,'blueWins_posthoc':d.blueWins},r/'cluster_assignments.csv')
    save_csv([{'seed':seed,'adjusted_rand_index':adjusted_rand_score(lab,KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(z))} for seed in [7,21,84]],r/'cluster_stability.csv')
    plt.figure(figsize=(9,3.5));plt.imshow(model.cluster_centers_,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto');plt.xticks(range(c.shape[1]),c.columns,rotation=35,ha='right',fontsize=8);plt.yticks(range(k),[f'Cụm {i}' for i in range(k)]);plt.colorbar(label='Trung tâm (z-score)');save_fig(f/'cluster_profiles.png')
    pca=PCA(n_components=2,random_state=SEED);pc=pca.fit_transform(z)
    plt.figure(figsize=(6,4))
    for cluster_id in range(k):
        mask=lab==cluster_id;plt.scatter(pc[mask,0],pc[mask,1],alpha=.35,s=6,label=f'Cụm {cluster_id}')
    plt.legend(markerscale=2);plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})');plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})');save_fig(f/'cluster_pca.png')
    save_json({'selected_k':k,'criterion':'maximum silhouette on fixed Train sample; elbow descriptive only',
      'silhouette_sample':float(ks.loc[ks.k==k,'silhouette_sample'].iloc[0]),'sample_n':len(sample),'fit_n':len(d),
      'features':list(c),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'blueWins':'posthoc description only; never used for K choice'},r/'clustering_metadata.json')
    return k

def build_transactions(d,threshold):
    x=feature_engineering(d)['difference'];items={}
    for side,sign in [('Blue',1),('Red',-1)]:
        for label,c in [('Gold_Lead','blueGoldDiff'),('Experience_Lead','blueExperienceDiff'),
         ('Kill_Lead','killDiff'),('CS_Lead','minionDiff'),('Level_Lead','avgLevelDiff'),('Tower_Lead','towerDiff')]:items[side+'_'+label]=sign*x[c]>0
        items[side+'_Large_Gold_Lead']=sign*x.blueGoldDiff>=threshold
        for label,c in [('Dragon','Dragons'),('Herald','Heralds'),('FirstBlood','FirstBlood')]:items[side+'_'+label]=d[side.lower()+c]>0
        items[side+'_Win']=d.blueWins==(1 if side=='Blue' else 0)
    return pd.DataFrame(items,index=d.index).astype(bool)

def apriori(t,min_support=.05,max_len=3):
    """Apriori theo tầng, phép nối + tỉa mọi tập con, giới hạn độ dài 3.
    Do đó antecedent tối đa 2 item khi consequent là một nhãn thắng.
    """
    masks={c:t[c].to_numpy() for c in t};freq={}
    level={frozenset([c]):float(t[c].mean()) for c in t if t[c].mean()>=min_support};freq.update(level)
    for size in range(2,max_len+1):
        prev=set(level);candidates=set();ls=sorted(prev,key=lambda a:tuple(sorted(a)))
        for i,a in enumerate(ls):
            for b in ls[i+1:]:
                u=a|b
                if len(u)==size and all(frozenset(v) in prev for v in itertools.combinations(sorted(u),size-1)):candidates.add(u)
        level={}
        for cand in sorted(candidates,key=lambda a:tuple(sorted(a))):
            support=float(np.logical_and.reduce([masks[c] for c in sorted(cand)]).mean())
            if support>=min_support:level[cand]=support
        freq.update(level)
        if not level:break
    return freq

def perform_association_rules(train,val,r,f):
    threshold=float(train.blueGoldDiff.abs().quantile(.75))
    t=build_transactions(train,threshold);v=build_transactions(val,threshold);freq=apriori(t)
    save_json({'large_gold_lead':threshold,'threshold_source':'75th percentile of abs blueGoldDiff on Train',
      'min_support':.05,'min_confidence':.70,'min_lift_exclusive':1.0,'max_itemset_length':3,
      'max_antecedent_length':2,'representative_rule_selection':'top 6 each winner by Train lift, confidence, support; exclude nested gold lead pair'},r/'association_metadata.json')
    save_csv([{'item':c,'train_support':t[c].mean(),'definition':('blueWins = 1' if c=='Blue_Win' else 'blueWins = 0' if c=='Red_Win' else c)} for c in t],r/'transaction_items.csv')
    save_csv([{'itemset':' & '.join(sorted(k)),'length':len(k),'support':val} for k,val in sorted(freq.items(),key=lambda kv:(len(kv[0]),sorted(kv[0])))],r/'frequent_itemsets.csv')
    rows=[]
    for its,support in freq.items():
        for target in ['Blue_Win','Red_Win']:
            if target not in its or len(its)<2:continue
            ant=its-{target}
            if ant & {'Blue_Win','Red_Win'}:continue
            confidence=support/freq[frozenset(ant)];lift=confidence/freq[frozenset([target])]
            if confidence<.70 or lift<=1:continue
            cols=sorted(ant);mask=v[cols].all(axis=1);an=int(mask.sum());both=int((mask & v[target]).sum());vc=both/an if an else np.nan
            rows.append({'antecedent':' & '.join(cols),'consequent':target,'antecedent_len':len(cols),
             'support_train':support,'confidence_train':confidence,'lift_train':lift,
             'antecedent_n_train':int(t[cols].all(axis=1).sum()),'joint_n_train':int((t[cols].all(axis=1)&t[target]).sum()),
             'antecedent_n_validation':an,'joint_n_validation':both,'support_validation':both/len(v),
             'confidence_validation':vc,'lift_validation':vc/v[target].mean()})
    rules=pd.DataFrame(rows).sort_values(['lift_train','confidence_train','support_train','antecedent'],ascending=[False,False,False,True])
    save_csv(rules,r/'association_rules.csv')
    # Không loại luật khỏi tập hợp lệ; chỉ bớt luật bao hàm hiển nhiên trong bảng tiêu biểu.
    eligible=rules[~rules.antecedent.str.contains(r'(?:Blue_Gold_Lead.*Blue_Large_Gold_Lead|Red_Gold_Lead.*Red_Large_Gold_Lead)',regex=True)]
    top=pd.concat([eligible[eligible.consequent==target].head(6) for target in ['Blue_Win','Red_Win']]).reset_index(drop=True)
    top.insert(0,'rule_id',[f'R{i+1:02d}' for i in range(len(top))]);save_csv(top,r/'association_rules_top.csv')
    plt.figure(figsize=(7,4));a=top.iloc[::-1];yp=np.arange(len(a));plt.barh(yp-.18,a.confidence_train,height=.35,label='Train');plt.barh(yp+.18,a.confidence_validation,height=.35,label='Validation');plt.yticks(yp,a.rule_id);plt.xlim(0,1);plt.xlabel('Confidence');plt.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);save_fig(f/'association_rules.png')
    return len(freq),len(rules)

def main():
    parser=argparse.ArgumentParser(description=__doc__);base=Path(__file__).resolve().parent
    parser.add_argument('--data',type=Path,default=base/'data/high_diamond_ranked_10min.csv')
    parser.add_argument('--output',type=Path,default=base)
    args=parser.parse_args();out=args.output.resolve();r=out/'results';f=out/'figures'
    for p in [r,f]:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(r/'run.log',mode='w',encoding='utf-8'),logging.StreamHandler()])
    started=time.perf_counter();logging.info('Bắt đầu; CSV=%s',args.data)
    with threadpool_limits(limits=2):
        d=load_data(args.data);summary=inspect_data(d,r);audit_redundancy(d,r);s=split_data(d,r)
        features=feature_engineering(d)
        save_json({k:list(v) for k,v in features.items()},r/'feature_sets.json')
        perform_eda(d.loc[s['Train']],r,f)
        winner,best,grids=train_classification_models(features,d,s,r,f)
        perform_ablation(features['difference'],d,s,r,f)
        scores=evaluate_test(winner,best,features,d,s,r,f)
        k=perform_clustering(d.loc[s['Train']],r,f)
        n_item,n_rules=perform_association_rules(d.loc[s['Train']],d.loc[s['Validation']],r,f)
    manifest={'status':'completed','utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
     'python':platform.python_version(),'versions':{'numpy':np.__version__,'pandas':pd.__version__,
      'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'joblib':joblib.__version__},
     'dataset_sha256':sha(args.data),'code_sha256':sha(Path(__file__)),
     'data_summary':summary,'initial_predictors':38,'reduced_raw_features':features['reduced_raw'].shape[1],
     'difference_features':features['difference'].shape[1],'selected':winner,'test_metrics':scores,
     'selected_k':k,'frequent_itemsets_max_len3':n_item,'filtered_rules':n_rules,
     'elapsed_seconds':time.perf_counter()-started}
    save_json(manifest,r/'run_manifest.json');logging.info('HOÀN TẤT: %s',json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
