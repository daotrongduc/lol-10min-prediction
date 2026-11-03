"""Web Streamlit: suy luận từ mô hình đã lưu, không huấn luyện trong ứng dụng."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import csv
import hashlib
import io
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
from league_of_legends_10min_data_mining import SUFFIXES, DROP_REDUNDANT, DESCRIPTIONS, feature_engineering

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
RAW_COLUMNS = [side + suffix for side in ('blue', 'red') for suffix in SUFFIXES
               if side + suffix not in DROP_REDUNDANT]
FULL_COLUMNS = [side + suffix for side in ('blue', 'red') for suffix in SUFFIXES]
PAGES = ['Dự đoán một trận', 'Dự đoán từ CSV', 'Kết quả thực nghiệm', 'Phân cụm & luật', 'Hướng dẫn']
GROUPS = [
    ('Kinh tế', ['TotalGold']),
    ('Giao tranh', ['Kills', 'Assists']),
    ('Kinh nghiệm & cấp độ', ['TotalExperience', 'AvgLevel']),
    ('Lính & quái rừng', ['TotalMinionsKilled', 'TotalJungleMinionsKilled']),
    ('Mục tiêu lớn', ['Dragons', 'Heralds', 'TowersDestroyed']),
    ('Tầm nhìn', ['WardsPlaced', 'WardsDestroyed']),
]
MAX_ROWS = 10000
MAX_BYTES = 10 * 1024 * 1024


class InputError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('; '.join(i['Lỗi'] for i in issues[:3]))


def issue(column, text, row='—'):
    return {'Dòng dữ liệu': row, 'Cột': column, 'Lỗi': text}


def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    """CSV dấu phẩy, UTF-8/BOM; giữ ID dạng chuỗi, phát hiện header trùng."""
    if len(payload) > MAX_BYTES:
        raise InputError([issue('Tệp', 'Dung lượng tối đa 10 MB.')])
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise InputError([issue('Tệp', 'Hãy lưu CSV ở định dạng UTF-8, phân cách bằng dấu phẩy.')]) from exc
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise InputError([issue('Tệp', 'Tệp rỗng hoặc tiêu đề CSV không hợp lệ.')]) from exc
    if len(header) != len(set(header)):
        raise InputError([issue('Tiêu đề', 'Có tên cột trùng. Mỗi cột chỉ được xuất hiện một lần.')])
    try:
        frame = pd.read_csv(io.StringIO(text), dtype={'gameId': 'string'})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise InputError([issue('Tệp', 'Không đọc được CSV. Kiểm tra số cột, dấu phẩy và dấu ngoặc kép.')]) from exc
    if len(frame) == 0 or len(frame) > MAX_ROWS:
        raise InputError([issue('Tệp', f'Cần từ 1 đến {MAX_ROWS:,} dòng dữ liệu.')])
    return frame.reset_index(drop=True)


def prepare_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Đủ 25 biến gốc -> 38 biến nhất quán -> dùng hàm gốc. Không bù dữ liệu thiếu."""
    if frame.empty or len(frame) > MAX_ROWS:
        raise InputError([issue('Tệp', f'Cần từ 1 đến {MAX_ROWS:,} dòng dữ liệu.')])
    if frame.columns.duplicated().any():
        raise InputError([issue('Tiêu đề', 'Có tên cột trùng.')])
    missing = [c for c in RAW_COLUMNS if c not in frame]
    if missing:
        raise InputError([issue(c, 'Thiếu cột bắt buộc.') for c in missing])
    issues = []
    def flag(mask, col, msg):
        for pos in np.flatnonzero(np.asarray(mask))[:20]:
            issues.append(issue(col, msg, int(pos) + 1))
    raw = pd.DataFrame(index=range(len(frame)))
    for col in RAW_COLUMNS:
        values = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)
        raw[col] = values
        finite = np.isfinite(values)
        flag(~finite, col, 'Cần một số hữu hạn; không để trống.')
        flag(finite & (values < 0), col, 'Không được âm.')
        if not col.endswith('AvgLevel'):
            flag(finite & (values != np.floor(values)), col, 'Cần số nguyên.')
        if col.endswith(('FirstBlood', 'Dragons', 'Heralds')):
            flag(finite & ~np.isin(values, [0, 1]), col, 'Schema phút 10 của mô hình chỉ hỗ trợ 0 hoặc 1.')
        if col.endswith('AvgLevel'):
            flag(finite & ((values < 1) | (values > 18)), col, 'Cấp độ trung bình phải nằm trong khoảng 1–18.')
    if issues:
        raise InputError(issues)
    for side in ('blue', 'red'):
        flag((raw[side + 'Kills'] == 0) & (raw[side + 'Assists'] != 0), side + 'Assists',
             'Đội chưa hạ gục đối thủ thì số hỗ trợ hạ gục phải bằng 0.')
    flag((raw.blueFirstBlood == 1) & (raw.blueKills < 1), 'blueFirstBlood', 'Blue có First Blood thì Blue phải có ít nhất 1 mạng.')
    flag((raw.blueFirstBlood == 0) & (raw.redKills < 1), 'blueFirstBlood', 'Trong schema này, 0 nghĩa là Red có First Blood; Red phải có ít nhất 1 mạng.')
    full = raw.copy()
    full['redDeaths'] = raw.blueKills
    full['blueDeaths'] = raw.redKills
    full['redFirstBlood'] = 1 - raw.blueFirstBlood
    full['blueGoldDiff'] = raw.blueTotalGold - raw.redTotalGold
    full['redGoldDiff'] = -full.blueGoldDiff
    full['blueExperienceDiff'] = raw.blueTotalExperience - raw.redTotalExperience
    full['redExperienceDiff'] = -full.blueExperienceDiff
    for side in ('blue', 'red'):
        full[side + 'CSPerMin'] = raw[side + 'TotalMinionsKilled'] / 10
        full[side + 'GoldPerMin'] = raw[side + 'TotalGold'] / 10
        full[side + 'EliteMonsters'] = raw[side + 'Dragons'] + raw[side + 'Heralds']
    # Nếu CSV cung cấp biến dẫn xuất, kiểm tra thay vì âm thầm ghi đè giá trị sai.
    for col in DROP_REDUNDANT:
        if col in frame:
            supplied = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)
            good = np.isfinite(supplied) & np.isclose(supplied, full[col], rtol=0, atol=1e-8)
            flag(~good, col, 'Không khớp công thức suy ra từ các cột gốc, hoặc giá trị bị thiếu/sai.')
    if issues:
        raise InputError(issues)
    ignored = [str(c) for c in frame if c not in FULL_COLUMNS + ['gameId', 'blueWins']]
    return full[FULL_COLUMNS], ignored


@st.cache_resource(show_spinner=False)
def load_artifact(model_mtime: int):
    artifact = joblib.load(RESULTS / 'final_model.joblib')
    required = {'model', 'feature_set', 'feature_names', 'threshold'}
    if not isinstance(artifact, dict) or not required <= artifact.keys():
        raise ValueError('Cấu trúc mô hình không đúng artifact của dự án.')
    names = artifact['feature_names']
    if artifact['feature_set'] not in ('reduced_raw', 'difference') or not names:
        raise ValueError('Bộ đặc trưng chưa được hỗ trợ.')
    if {'gameId', 'blueWins'} & set(names) or len(names) != len(set(names)):
        raise ValueError('Danh sách đặc trưng không hợp lệ.')
    if not 0 < float(artifact['threshold']) < 1:
        raise ValueError('Ngưỡng mô hình không hợp lệ.')
    model = artifact['model']
    if set(model.classes_) != {0, 1} or model.n_features_in_ != len(names):
        raise ValueError('Số biến hoặc quy ước nhãn của mô hình không đúng.')
    if hasattr(model, 'feature_names_in_') and list(model.feature_names_in_) != list(names):
        raise ValueError('Thứ tự biến trong artifact khác mô hình.')
    return artifact, threading.Lock()


def predict_frame(frame: pd.DataFrame, artifact, lock=None):
    full, ignored = prepare_input(frame)
    features = feature_engineering(full)[artifact['feature_set']]
    x = features.loc[:, artifact['feature_names']]
    # Chỉ suy luận. Khóa tài nguyên dùng chung khi có nhiều phiên truy cập.
    with (lock or threading.Lock()):
        with threadpool_limits(limits=2):
            probabilities = artifact['model'].predict_proba(x)
    classes = list(artifact['model'].classes_)
    blue = probabilities[:, classes.index(1)]
    red = probabilities[:, classes.index(0)]
    if not (np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
            and np.allclose(blue + red, 1)):
        raise ValueError('Mô hình trả về xác suất không hợp lệ.')
    pred = (blue >= artifact['threshold']).astype(int)
    output = pd.DataFrame({'prob_blue_win': blue, 'prob_red_win': red,
                           'prediction': pred, 'predicted_winner': np.where(pred == 1, 'Blue', 'Red')})
    if 'gameId' in frame:
        output.insert(0, 'gameId', frame.gameId.reset_index(drop=True))
    return output, full, ignored


def evaluate_uploaded_labels(frame, output):
    """Nhãn chỉ được đọc sau suy luận; nhãn lỗi không ngăn dự đoán hợp lệ."""
    y = pd.to_numeric(frame['blueWins'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isin(y, [0, 1])
    if not valid.all():
        raise InputError([issue('blueWins', 'Nhãn đối chiếu phải là 0 hoặc 1.', int(i) + 1)
                          for i in np.flatnonzero(~valid)[:20]])
    pred = output.prediction.to_numpy()
    metrics = {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
               'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
               'roc_auc': roc_auc_score(y, output.prob_blue_win) if len(np.unique(y)) == 2 else None}
    return metrics, y.astype(int)


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: int):
    return pd.read_csv(path)


def result_csv(name):
    path = RESULTS / name
    return load_csv(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: int):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def result_json(name):
    path = RESULTS / name
    return load_json(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_samples(mtime: int):
    return pd.read_csv(BASE / 'samples/test_matches.csv', dtype={'gameId': 'string'})


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def show_issues(exc):
    st.error('Vui lòng sửa dữ liệu trước khi dự đoán. Dòng 1 là dòng dữ liệu đầu tiên, không tính tiêu đề.')
    st.dataframe(pd.DataFrame(exc.issues), hide_index=True, width='stretch')
    st.caption('Hiển thị tối đa 20 dòng lỗi cho mỗi điều kiện kiểm tra.')


def show_metrics(scores):
    labels = [('accuracy', 'Accuracy'), ('precision', 'Precision · Blue'), ('recall', 'Recall · Blue'),
              ('f1', 'F1 · Blue'), ('roc_auc', 'ROC-AUC')]
    for container, (key, label) in zip(st.columns(5), labels):
        value = scores.get(key)
        container.metric(label, 'Không xác định' if value is None else (f'{value:.4f}' if key == 'roc_auc' else f'{value:.2%}'))


def sample_to_form(row):
    draft = {c: float(row[c]) if c.endswith('AvgLevel') else int(row[c]) for c in RAW_COLUMNS}
    st.session_state['draft'] = draft
    for c, value in draft.items():
        st.session_state['input_' + c] = value
    st.session_state['sample_origin'] = {'gameId': str(row.gameId), 'blueWins': int(row.blueWins), 'raw': draft.copy()}
    st.session_state.pop('single_result', None)


def clear_form():
    st.session_state['draft'] = {c: None for c in RAW_COLUMNS}
    for c in RAW_COLUMNS:
        st.session_state['input_' + c] = None
    st.session_state.pop('sample_origin', None)
    st.session_state.pop('single_result', None)


def range_warnings(full):
    profile = result_csv('input_ranges_train.csv').set_index('feature')
    messages = []
    for c in RAW_COLUMNS:
        if c in profile.index:
            mask = (full[c] < profile.loc[c, 'min']) | (full[c] > profile.loc[c, 'max'])
            if mask.any():
                messages.append(f'{c}: {int(mask.sum())} dòng ngoài khoảng quan sát ở Train')
    return messages


def single_page(artifact, lock):
    st.markdown('### Thử một trạng thái trận đấu')
    st.write('Nạp một trận Test để bắt đầu, hoặc xóa biểu mẫu và nhập thống kê của bạn.')
    path = BASE / 'samples/test_matches.csv'
    samples = load_samples(path.stat().st_mtime_ns)
    if 'draft' not in st.session_state:
        sample_to_form(samples.iloc[0])
    left, middle, right = st.columns([3, 1, 1])
    with left:
        selected = st.selectbox('Trận mẫu từ tập Test', samples.gameId.tolist(), key='sample_choice')
    with middle:
        st.button('Nạp trận mẫu', on_click=sample_to_form,
                  args=(samples.loc[samples.gameId.eq(selected)].iloc[0],), width='stretch', key='load_sample')
    with right:
        st.button('Xóa để nhập mới', on_click=clear_form, width='stretch', key='clear_form')
    origin = st.session_state.get('sample_origin')
    st.caption(f'{len(samples):,} trận Test, gồm cả dự đoán đúng và sai. '
               + (f'Biểu mẫu được nạp từ trận {origin["gameId"]}.' if origin else 'Đang nhập một tình huống mới.'))
    for c in RAW_COLUMNS:
        st.session_state.setdefault('input_' + c, st.session_state['draft'].get(c))
    with st.form('match_form', clear_on_submit=False):
        values = {}
        values['blueFirstBlood'] = st.selectbox('Đội giành First Blood (mạng hạ gục đầu tiên)',
            options=[1, 0], format_func=lambda x: 'Blue' if x == 1 else 'Red',
            index=None, key='input_blueFirstBlood', placeholder='Chọn đội')
        st.caption('Mô hình dùng schema có một đội đã giành First Blood; không hỗ trợ trường hợp chưa có First Blood.')
        blue_col, red_col = st.columns(2, gap='large')
        for side, container, title, color in [('blue', blue_col, 'ĐỘI BLUE', '#2477bd'), ('red', red_col, 'ĐỘI RED', '#b83b4d')]:
            with container:
                st.markdown(f'<div class="team-title" style="border-color:{color};color:{color}">{title}</div>', unsafe_allow_html=True)
                for label, suffixes in GROUPS:
                    st.markdown(f'**{label}**')
                    for suffix in suffixes:
                        col = side + suffix
                        desc, unit = DESCRIPTIONS[suffix]
                        opts = dict(label=f'{desc} ({unit})', key='input_' + col,
                                    help=f'{col} — thống kê của đội tại phút 10.', value=None)
                        if suffix == 'AvgLevel':
                            values[col] = st.number_input(**opts, min_value=1.0, max_value=18.0, step=0.2, format='%.2f')
                        elif suffix in ('Dragons', 'Heralds'):
                            values[col] = st.number_input(**opts, min_value=0, max_value=1, step=1)
                        else:
                            values[col] = st.number_input(**opts, min_value=0, step=1)
        submitted = st.form_submit_button('Dự đoán kết quả', type='primary', width='stretch')
    if submitted:
        st.session_state['draft'] = values.copy()
        st.session_state.pop('single_result', None)
        try:
            with st.spinner('Đang dự đoán từ thống kê phút 10…'):
                output, full, _ = predict_frame(pd.DataFrame([values]), artifact, lock)
            unchanged = bool(origin) and all(values[c] == origin['raw'][c] for c in RAW_COLUMNS)
            st.session_state['single_result'] = {
                'output': output.iloc[0].to_dict(), 'full': full.iloc[0].to_dict(),
                'origin': origin if unchanged else None, 'modified': bool(origin) and not unchanged,
                'warnings': range_warnings(full)}
        except InputError as exc:
            show_issues(exc)
        except Exception:
            st.error('Không thể dự đoán. Kiểm tra tệp mô hình và phiên bản thư viện theo README.')
    result = st.session_state.get('single_result')
    if result:
        st.divider()
        st.caption('Kết quả của lần bấm “Dự đoán kết quả” gần nhất. Sau khi sửa ô nhập, hãy bấm lại để cập nhật.')
        out = result['output']
        st.subheader(f'Đội được dự đoán thắng: {out["predicted_winner"]}')
        a, b = st.columns(2)
        a.metric('Xác suất Blue thắng', f'{out["prob_blue_win"]:.2%}')
        b.metric('Xác suất Red thắng', f'{out["prob_red_win"]:.2%}')
        st.progress(float(out['prob_blue_win']), text=f'Blue {out["prob_blue_win"]:.1%} · Red {out["prob_red_win"]:.1%}')
        st.caption('Đây là xác suất mô hình ước tính cho một trận, không phải Accuracy và không bảo đảm kết quả thực tế.')
        full = result['full']
        cols = st.columns(3)
        for box, label, value in [(cols[0], 'Vàng · Blue − Red', full['blueGoldDiff']),
                                  (cols[1], 'Kinh nghiệm · Blue − Red', full['blueExperienceDiff']),
                                  (cols[2], 'Mạng · Blue − Red', full['blueKills'] - full['redKills'])]:
            box.metric(label, f'{value:+,.0f}')
        st.caption('Các chênh lệch trên mô tả dữ liệu đầu vào; không phải giải thích nhân quả cho quyết định của mô hình.')
        if result['warnings']:
            st.warning('Một số giá trị nằm ngoài khoảng đã thấy trong Train; mô hình có thể dự đoán kém tin cậy hơn.')
            with st.expander('Xem các giá trị ngoài khoảng quan sát'): st.write(result['warnings'])
        if result['origin']:
            if st.checkbox('Hiện kết quả thực tế của trận mẫu', key='reveal_truth'):
                truth = result['origin']['blueWins']
                actual = 'Blue' if truth == 1 else 'Red'
                text = f'Trận {result["origin"]["gameId"]} · Thực tế: {actual} thắng · Dự đoán '
                if int(out['prediction']) == truth: st.success(text + 'đúng.')
                else: st.warning(text + 'sai.')
        elif result['modified']:
            st.info('Bạn đã sửa thông số của trận mẫu. Đây là tình huống giả định; không dùng kết quả trận gốc để chấm đúng/sai.')
        st.download_button('Tải kết quả trận này', csv_bytes(pd.DataFrame([out])), 'du_doan_mot_tran.csv', 'text/csv')


def batch_page(artifact, lock):
    st.markdown('### Dự đoán nhiều trận từ CSV')
    st.write('Tải dữ liệu phút 10, kiểm tra các cột rồi dự đoán cả bảng trong một lần.')
    sample = BASE / 'samples/input_template.csv'
    st.download_button('Tải CSV mẫu không có nhãn', sample.read_bytes(), 'mau_du_doan_lol.csv', 'text/csv')
    with st.expander('Các cột được chấp nhận'):
        st.write('Bắt buộc đủ 25 cột sau. CSV gốc có 38 cột thống kê cũng dùng được; các biến dẫn xuất sẽ được kiểm tra nhất quán.')
        st.code(','.join(RAW_COLUMNS), language='text')
        st.write('gameId và blueWins là tùy chọn. Nhãn chỉ dùng đối chiếu sau dự đoán. CSV UTF-8, phân cách dấu phẩy; tối đa 10 MB / 10.000 dòng.')
    uploaded = st.file_uploader('Chọn file CSV thống kê phút 10', type=['csv'], key='batch_upload')
    if uploaded is None:
        st.info('Chọn một tệp để xem trước dữ liệu và bắt đầu dự đoán.')
        st.session_state.pop('batch_result', None)
        return
    payload = uploaded.getvalue()
    signature = hashlib.sha256(payload).hexdigest()
    if st.session_state.get('batch_signature') != signature:
        st.session_state.pop('batch_result', None)
        st.session_state['batch_signature'] = signature
    try:
        frame = read_csv_bytes(payload)
    except InputError as exc:
        show_issues(exc); return
    st.caption(f'{len(frame):,} dòng · {len(frame.columns)} cột. Xem trước tối đa 10 dòng.')
    st.dataframe(frame.head(10), hide_index=True, width='stretch')
    if st.button('Dự đoán file CSV', type='primary', key='predict_batch'):
        st.session_state.pop('batch_result', None)
        try:
            with st.spinner('Đang kiểm tra dữ liệu và dự đoán…'):
                output, full, ignored = predict_frame(frame, artifact, lock)
            label_errors, metrics = [], None
            if 'blueWins' in frame:
                try:
                    metrics, truth = evaluate_uploaded_labels(frame, output)
                    output['actual_blueWins'] = truth
                    output['correct'] = output.prediction.to_numpy() == truth
                except InputError as exc:
                    label_errors = exc.issues
            st.session_state['batch_result'] = {'output': output, 'metrics': metrics,
                'label_errors': label_errors, 'ignored': ignored, 'warnings': range_warnings(full)}
        except InputError as exc:
            show_issues(exc)
        except Exception:
            st.error('Không thể dự đoán. Kiểm tra mô hình và môi trường theo README.')
    result = st.session_state.get('batch_result')
    if result:
        output = result['output']
        st.success(f'Đã dự đoán {len(output):,} trận.')
        if result['ignored']: st.info('Các cột bổ sung không dùng cho mô hình: ' + ', '.join(result['ignored']))
        if result['warnings']:
            st.warning('Có dữ liệu ngoài khoảng quan sát của Train. Đã dự đoán nhưng cần thận trọng khi diễn giải.')
            with st.expander('Chi tiết'): st.write(result['warnings'])
        st.dataframe(output, hide_index=True, width='stretch')
        st.download_button('Tải toàn bộ kết quả CSV', csv_bytes(output), 'ket_qua_du_doan_lol.csv', 'text/csv', key='download_batch')
        if result['label_errors']:
            st.warning('Dự đoán đã hoàn tất, nhưng nhãn đối chiếu không hợp lệ nên không tính chỉ số.')
            st.dataframe(pd.DataFrame(result['label_errors']), hide_index=True)
        if result['metrics']:
            st.markdown('#### Kết quả trên file vừa tải')
            show_metrics(result['metrics'])
            st.caption('Các chỉ số này phụ thuộc file tải lên; không thay thế kết quả Test chính thức. Precision/Recall/F1 dùng lớp Blue thắng.')
            if result['metrics']['roc_auc'] is None:
                st.info('Nhãn thực tế chỉ có một lớp nên ROC-AUC không xác định.')


def show_figure(name, caption):
    path = FIGURES / name
    if path.is_file(): st.image(str(path), caption=caption, width='stretch')
    else: st.info(f'Chưa có hình {name}. Hãy bổ sung thư mục figures từ lần thực nghiệm.')


def experiment_page():
    summary = result_csv('dataset_summary.csv').iloc[0]
    test = result_csv('classification_test.csv').iloc[0]
    selected = result_json('selected_model.json')
    st.markdown('### Kết quả thực nghiệm đã lưu')
    st.write('Dự đoán kết quả cuối trận bằng thống kê phút 10. Mô hình được chọn theo F1 Validation rồi refit trên Train + Validation.')
    a, b, c = st.columns(3)
    a.metric('Trận đấu', f'{int(summary["rows"]):,}')
    b.metric('Cột dữ liệu gốc', int(summary['columns']))
    c.metric('Blue thắng trong dữ liệu', f'{summary["blue_win_rate"]:.2%}')
    st.markdown('#### Phân chia dữ liệu')
    st.dataframe(result_csv('split_summary.csv').rename(columns={'split':'Tập','n':'Số trận','fraction':'Tỷ lệ','blue_wins':'Blue thắng','red_wins':'Red thắng','blue_win_rate':'Tỷ lệ Blue thắng'}), hide_index=True, width='stretch')
    st.markdown('#### Đánh giá trên Test')
    show_metrics(test.to_dict())
    st.write(f'**Mô hình:** {selected["model"]} · **Số đặc trưng:** {int(selected["n_features"])}')
    with st.expander('Cấu hình mô hình và quy trình đánh giá'):
        meta = result_json('final_model_metadata.json')
        st.json({'feature_set':meta['feature_set'],'threshold':meta['threshold'],'seed':meta['seed'],
                 'train_plus_validation_size':meta['train_plus_validation_size'],'test_size':meta['test_size'],
                 'parameters':meta['parameters']})
        st.write('gameId và blueWins không thuộc đầu vào. Không chọn lại mô hình theo điểm Test.')
    a, b = st.columns(2)
    with a: show_figure('confusion_matrix.png', 'Ma trận nhầm lẫn · Test')
    with b: show_figure('roc_curve.png', 'Đường ROC · Test')
    st.markdown('#### So sánh trên Validation')
    st.dataframe(result_csv('classification_validation.csv')[['model','feature_set','accuracy','precision','recall','f1','roc_auc']], hide_index=True, width='stretch')
    show_figure('model_comparison.png', 'F1 Validation tốt nhất của từng họ mô hình')
    st.markdown('#### Độ quan trọng và nhóm thông tin')
    st.caption('Permutation Importance tính trên Validation với mô hình chỉ fit Train, trước khi refit; không phải quan hệ nhân quả.')
    show_figure('feature_importance.png', 'Mức giảm F1 khi hoán vị đặc trưng')
    with st.expander('Xem bảng importance và ablation'):
        st.dataframe(result_csv('feature_importance.csv'), hide_index=True, width='stretch')
        st.dataframe(result_csv('validation_ablation.csv'), hide_index=True, width='stretch')
    with st.expander('Khám phá dữ liệu Train'):
        for name, caption in [('target_distribution.png','Phân phối nhãn'),('gold_diff_by_winner.png','Chênh lệch vàng theo đội thắng'),('correlation_heatmap.png','Tương quan Pearson')]:
            show_figure(name, caption)


def mining_page():
    st.markdown('### Cấu trúc trận đấu & các luật quan sát')
    st.write('Kết quả nghiên cứu trên Train; luật được kiểm chứng thêm trên Validation.')
    cluster_tab, rule_tab = st.tabs(['Phân cụm K-Means', 'Luật kết hợp Apriori'])
    with cluster_tab:
        meta = result_json('clustering_metadata.json')
        a, b, c = st.columns(3)
        a.metric('K được chọn', meta['selected_k'])
        b.metric('Silhouette', f'{meta["silhouette_sample"]:.4f}')
        c.metric('Số trận fit', meta['fit_n'])
        st.caption(f'Silhouette tính trên mẫu cố định {meta["sample_n"]:,} trận Train. PCA chỉ dùng minh họa; cụm không xác định đội nào thắng.')
        st.dataframe(result_csv('cluster_summary.csv'), hide_index=True, width='stretch')
        a, b = st.columns(2)
        with a: show_figure('elbow_curve.png','Elbow · Train')
        with b: show_figure('silhouette_scores.png','Silhouette · Train')
        show_figure('cluster_pca.png','Hình chiếu PCA của các cụm trong không gian đặc trưng')
        st.info('Đây là kết quả phân cụm đã lưu. Ứng dụng không gán cụm cho trận mới.')
    with rule_tab:
        manifest = result_json('run_manifest.json')
        meta = result_json('association_metadata.json')
        a, b = st.columns(2)
        a.metric('Tập mục phổ biến (độ dài ≤ 3)', manifest['frequent_itemsets_max_len3'])
        b.metric('Luật sau lọc', manifest['filtered_rules'])
        st.write(f'Ngưỡng: Support ≥ {meta["min_support"]:.0%}; Confidence ≥ {meta["min_confidence"]:.0%}; Lift > {meta["min_lift_exclusive"]:g}. '
                 f'Lợi thế vàng lớn: từ {meta["large_gold_lead"]:,.1f} vàng theo phân vị Train.')
        st.caption('Support: tỷ lệ trận thỏa cả điều kiện và kết quả. Confidence: tỷ lệ thắng khi thỏa điều kiện. Lift: confidence chia tỷ lệ thắng nền.')
        st.dataframe(result_csv('association_rules_top.csv')[['rule_id','antecedent','consequent','support_train','confidence_train','lift_train','antecedent_n_validation','confidence_validation']], hide_index=True, width='stretch')
        show_figure('association_rules.png','Confidence của cùng các luật trên Train và Validation')
        st.caption('Luật mô tả liên hệ quan sát; confidence cao không bảo đảm thắng và không chứng minh nhân quả.')
        rules = result_csv('association_rules.csv')
        st.download_button('Tải toàn bộ luật kết hợp', csv_bytes(rules), 'luat_ket_hop.csv', 'text/csv')


def guide_page():
    st.markdown('### Dùng ứng dụng trong buổi trình diễn')
    st.markdown('''1. Mở **Dự đoán một trận**, chọn mã trận Test và bấm **Nạp trận mẫu**.
2. Bấm **Dự đoán kết quả**, rồi chọn **Hiện kết quả thực tế** để đối chiếu.
3. Thay đổi vàng hoặc kinh nghiệm, bấm dự đoán lại để thử tình huống giả định.
4. Mở **Dự đoán từ CSV**, tải CSV mẫu, chọn tệp và chạy dự đoán theo lô.
5. Mở **Kết quả thực nghiệm** và **Phân cụm & luật** để trình bày phương pháp và kết quả.''')
    st.info('Đầu vào luôn là thống kê tại phút thứ 10; đầu ra là kết quả cuối trận. Không nhập dữ liệu tổng kết sau trận để dùng mô hình này.')
    st.markdown('#### Những điều cần phân biệt')
    st.write('Xác suất của một trận không phải Accuracy. Accuracy là tỷ lệ dự đoán đúng trên một tập nhiều trận đã có nhãn. Mô hình có thể dự đoán sai dù xác suất cao.')
    st.write('Schema gốc chỉ có trường hợp Blue hoặc Red giành First Blood; không dùng ứng dụng này để suy luận trường hợp chưa có First Blood. Rồng và sứ giả dùng mã 0/1 theo dữ liệu gốc.')
    st.write('CSV được xử lý trong phiên ứng dụng và không được ghi ra đĩa bởi code web. Nút tải xuống cho phép bạn tự lưu kết quả.')
    st.markdown('#### Tài liệu triển khai')
    st.markdown('[Hướng dẫn Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)')
    st.caption('Hướng dẫn cài đặt Python 3.12, chạy trên máy và triển khai GitHub → Streamlit có trong README của dự án.')


def main():
    st.set_page_config(page_title='LoL · Dự đoán phút 10', page_icon='⚔️', layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1180px; padding-top:2.1rem; padding-bottom:3rem;}
    h1 {letter-spacing:-.035em; font-weight:750;}
    .eyebrow {font-size:.76rem; letter-spacing:.16em; font-weight:700; color:#66768a; margin-bottom:.5rem;}
    .team-title {border-top:3px solid; padding:.8rem 0; font-size:1.1rem; font-weight:750; letter-spacing:.08em; margin-bottom:.8rem;}
    [data-testid="stMetric"] {background:#f1f5f9; border:1px solid #e2e8f0; border-radius:12px; padding:14px;}
    [data-testid="stSidebar"] {border-right:1px solid #e2e8f0;}
    [data-testid="stForm"] {border-radius:16px; padding:1.5rem;}
    @media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem;} h1{font-size:1.9rem;}}
    </style>''', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('## LoL · Phút 10')
        st.caption('KHAI PHÁ DỮ LIỆU TRẬN ĐẤU')
        page = st.radio('Điều hướng', PAGES, label_visibility='collapsed', key='navigation')
        st.divider()
        st.caption('THỜI ĐIỂM DỮ LIỆU')
        st.markdown('**10 phút đầu trận**')
        st.caption('Dự đoán Blue / Red thắng cuối trận từ mô hình đã huấn luyện.')
    st.markdown('<div class="eyebrow">LEAGUE OF LEGENDS / PHÂN TÍCH TRẬN ĐẤU</div>', unsafe_allow_html=True)
    st.title('Dự đoán kết quả trận đấu LoL')
    st.markdown('Dựa trên thống kê của hai đội tại phút thứ 10')
    st.divider()
    try:
        if page in PAGES[:2]:
            path = RESULTS / 'final_model.joblib'
            if not path.is_file():
                st.error('Chưa tìm thấy results/final_model.joblib. Hãy bổ sung mô hình trong gói dự án hoặc chạy file thực nghiệm một lần trên máy, rồi khởi động lại web.')
                st.stop()
            with st.spinner('Đang nạp mô hình…'):
                artifact, lock = load_artifact(path.stat().st_mtime_ns)
            if page == PAGES[0]: single_page(artifact, lock)
            else: batch_page(artifact, lock)
        elif page == PAGES[2]: experiment_page()
        elif page == PAGES[3]: mining_page()
        else: guide_page()
    except FileNotFoundError as exc:
        st.error(f'Thiếu tệp {Path(exc.filename).name if exc.filename else "kết quả"}. Hãy giữ nguyên các thư mục results, figures và samples trong gói dự án.')
    except Exception:
        st.error('Ứng dụng chưa đọc được mô hình hoặc kết quả. Kiểm tra đủ tệp và cài đúng requirements.txt theo README; xem nhật ký máy chủ để chẩn đoán.')
        import logging
        logging.exception('Lỗi đọc mô hình/kết quả')
    st.divider()
    st.caption('Bài tập lớn · Khai phá dữ liệu | Kết quả phục vụ nghiên cứu và trình diễn.')


if __name__ == '__main__':
    main()
