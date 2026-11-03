"""Kiểm thử suy luận và luồng Streamlit; không gọi huấn luyện."""
from pathlib import Path
import io
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from streamlit.testing.v1 import AppTest
import app
from league_of_legends_10min_data_mining import feature_engineering


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / 'samples/test_matches.csv')
        cls.artifact = joblib.load(ROOT / 'results/final_model.joblib')
        cls.small = cls.data.head(5).copy()

    def test_all_original_rows_pass_validation(self):
        full = pd.read_csv(ROOT / 'data/high_diamond_ranked_10min.csv')
        prepared, _ = app.prepare_input(full)
        self.assertEqual(len(prepared), 9879)
        self.assertTrue(np.isfinite(prepared.to_numpy()).all())

    def test_exact_predictions_vs_saved_test(self):
        pred, _, _ = app.predict_frame(self.data, self.artifact)
        saved = pd.read_csv(ROOT / 'results/test_predictions.csv')
        both = pred.merge(saved, on='gameId', validate='one_to_one', suffixes=('_web', '_saved'))
        self.assertEqual(len(both), len(saved))
        np.testing.assert_allclose(both.prob_blue_win_web, both.prob_blue_win_saved, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(both.prediction, both.y_pred)

    def test_raw_25_full_38_and_direct_model_match(self):
        raw = self.small[app.RAW_COLUMNS]
        a, _, _ = app.predict_frame(raw, self.artifact)
        b, _, _ = app.predict_frame(self.small, self.artifact)
        np.testing.assert_array_equal(a.prob_blue_win, b.prob_blue_win)
        x = feature_engineering(self.small)[self.artifact['feature_set']][self.artifact['feature_names']]
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2): direct = self.artifact['model'].predict_proba(x)[:, 1]
        np.testing.assert_array_equal(a.prob_blue_win, direct)

    def test_labels_and_ids_do_not_change_prediction(self):
        changed = self.small.copy()
        changed['blueWins'] = 1 - changed.blueWins
        changed['gameId'] = 'arbitrary-id'
        a, _, _ = app.predict_frame(self.small, self.artifact)
        b, _, _ = app.predict_frame(changed, self.artifact)
        np.testing.assert_array_equal(a.prob_blue_win, b.prob_blue_win)
        self.assertFalse(set(self.artifact['feature_names']) & {'blueWins', 'gameId'})

    def test_single_equals_batch_and_probabilities_valid(self):
        batch, _, _ = app.predict_frame(self.small, self.artifact)
        for i in range(len(self.small)):
            single, _, _ = app.predict_frame(self.small.iloc[[i]], self.artifact)
            self.assertEqual(single.prob_blue_win.iloc[0], batch.prob_blue_win.iloc[i])
        np.testing.assert_allclose(batch.prob_blue_win + batch.prob_red_win, 1)
        self.assertTrue(batch.prob_blue_win.between(0, 1).all())
        np.testing.assert_array_equal(batch.prediction, (batch.prob_blue_win >= self.artifact['threshold']).astype(int))

    def test_missing_invalid_fractional_negative_inputs_rejected(self):
        bad = self.small[app.RAW_COLUMNS].copy()
        with self.assertRaises(app.InputError): app.prepare_input(bad.drop(columns='blueTotalGold'))
        for col, value in [('blueKills', 'oops'), ('blueKills', np.nan), ('blueKills', np.inf),
                           ('blueKills', -1), ('blueKills', 2.5), ('blueAvgLevel', 19),
                           ('redDragons', 2), ('blueFirstBlood', 3)]:
            with self.subTest(col=col, value=value):
                case = bad.astype(object).copy(); case.loc[0, col] = value
                with self.assertRaises(app.InputError) as cm: app.prepare_input(case)
                self.assertEqual(cm.exception.issues[0]['Cột'], col)
                self.assertEqual(cm.exception.issues[0]['Dòng dữ liệu'], 1)

    def test_inconsistent_derived_and_first_blood_rejected(self):
        bad = self.small.copy(); bad.loc[0, 'blueGoldDiff'] += 1
        with self.assertRaises(app.InputError): app.prepare_input(bad)
        bad = self.small[app.RAW_COLUMNS].copy(); bad.loc[0, ['blueFirstBlood','blueKills','blueAssists']] = [1,0,0]
        with self.assertRaises(app.InputError): app.prepare_input(bad)

    def test_no_arbitrary_training_range_cap(self):
        case = self.small[app.RAW_COLUMNS].copy(); case.loc[0, 'blueTotalGold'] = 1000000
        out, _, _ = app.predict_frame(case, self.artifact)
        self.assertEqual(len(out), len(case))

    def test_parser_and_unicode_ids(self):
        raw = self.small[app.RAW_COLUMNS].copy(); raw.insert(0,'gameId',['00001','trận-2','3','4','5'])
        parsed = app.read_csv_bytes(app.csv_bytes(raw))
        self.assertEqual(parsed.gameId.iloc[0], '00001')
        self.assertEqual(parsed.gameId.iloc[1], 'trận-2')
        with self.assertRaises(app.InputError): app.read_csv_bytes(b'a,a\n1,2')
        with self.assertRaises(app.InputError): app.read_csv_bytes(b'')
        with self.assertRaises(app.InputError): app.read_csv_bytes(b'\xff')
        with self.assertRaises(app.InputError): app.read_csv_bytes(b'x' * (app.MAX_BYTES + 1))
        with self.assertRaises(app.InputError): app.prepare_input(pd.concat([raw] * 2001, ignore_index=True))

    def test_one_class_auc_and_optional_invalid_labels(self):
        one = self.small.copy(); one['blueWins'] = 1
        pred, _, _ = app.predict_frame(one, self.artifact)
        metrics, _ = app.evaluate_uploaded_labels(one, pred)
        self.assertIsNone(metrics['roc_auc'])
        one['blueWins'] = 'wrong'
        out, _, _ = app.predict_frame(one, self.artifact)
        self.assertEqual(len(out), len(one))
        with self.assertRaises(app.InputError): app.evaluate_uploaded_labels(one, out)

    def test_sample_set_complete_including_wrong_predictions(self):
        split = pd.read_csv(ROOT/'results/split_assignments.csv')
        expected = set(split.loc[split['split'].eq('Test'), 'gameId'])
        self.assertEqual(set(self.data.gameId), expected)
        saved = pd.read_csv(ROOT/'results/test_predictions.csv')
        self.assertTrue((saved.y_true == saved.y_pred).any())
        self.assertTrue((saved.y_true != saved.y_pred).any())


class InterfaceTests(unittest.TestCase):
    def new_app(self):
        return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()

    def assert_clean(self, at):
        self.assertEqual([e.message for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])

    def submit(self, at):
        return next(b for b in at.button if b.label == 'Dự đoán kết quả').click().run()

    def test_load_predict_reveal_modify_and_manual_entry(self):
        at = self.new_app(); self.assert_clean(at)
        self.assertEqual(len(at.number_input), 24)
        self.submit(at); self.assert_clean(at)
        before = at.session_state['single_result']
        self.assertIsNotNone(before['origin'])
        at.checkbox(key='reveal_truth').check().run(); self.assert_clean(at)
        origin = before['origin']; values = origin['raw']
        at.number_input(key='input_blueTotalGold').set_value(values['blueTotalGold']+300)
        self.submit(at); self.assert_clean(at)
        self.assertIsNone(at.session_state['single_result']['origin'])
        self.assertTrue(at.session_state['single_result']['modified'])
        self.assertTrue(any('giả định' in e.value for e in at.info))
        at.button(key='clear_form').click().run(); self.assert_clean(at)
        self.assertTrue(all(n.value is None for n in at.number_input))
        self.submit(at)
        self.assertTrue(at.error)
        for c in app.RAW_COLUMNS:
            if c != 'blueFirstBlood':at.number_input(key='input_'+c).set_value(values[c])
        at.selectbox(key='input_blueFirstBlood').select(values['blueFirstBlood'])
        self.submit(at); self.assert_clean(at)
        self.assertIsNone(at.session_state['single_result']['origin'])
        self.assertAlmostEqual(at.session_state['single_result']['output']['prob_blue_win'], before['output']['prob_blue_win'])
        options = at.selectbox(key='sample_choice').options
        at.selectbox(key='sample_choice').select(options[1]).run()
        at.button(key='load_sample').click().run(); self.submit(at); self.assert_clean(at)
        self.assertEqual(at.session_state['single_result']['origin']['gameId'], options[1])

    def test_all_pages_and_no_training_on_rerun(self):
        with patch.object(HistGradientBoostingClassifier, 'fit', side_effect=AssertionError('Web called fit')):
            at = self.new_app()
            for page in app.PAGES:
                at.radio(key='navigation').set_value(page).run(); self.assert_clean(at)
            at.radio(key='navigation').set_value(app.PAGES[0]).run()
            self.submit(at); self.assert_clean(at)
            at.run(); self.assert_clean(at)

    def test_csv_upload_flow_and_stale_result_clearing(self):
        raw = pd.read_csv(ROOT/'samples/input_template.csv', dtype={'gameId':'string'})
        raw['blueWins'] = 1
        upload = io.BytesIO(app.csv_bytes(raw)); upload.name = 'example.csv'
        with patch('streamlit.file_uploader', return_value=upload):
            at = self.new_app(); at.radio(key='navigation').set_value(app.PAGES[1]).run(); self.assert_clean(at)
            at.button(key='predict_batch').click().run(); self.assert_clean(at)
            result = at.session_state['batch_result']
            self.assertEqual(len(result['output']), 5)
            self.assertIsNone(result['metrics']['roc_auc'])
            self.assertTrue(any('một lớp' in i.value for i in at.info))
        bad = raw.copy();bad.loc[0,'blueKills'] = -1
        upload2 = io.BytesIO(app.csv_bytes(bad));upload2.name = 'invalid.csv'
        with patch('streamlit.file_uploader', return_value=upload2):
            at.run(); self.assertNotIn('batch_result', at.session_state)
            at.button(key='predict_batch').click().run()
            self.assertTrue(at.error); self.assertNotIn('batch_result', at.session_state)
            self.assertFalse(at.exception)

    def test_missing_model_has_actionable_error(self):
        original = Path.is_file
        def fake(path):
            return False if path.name == 'final_model.joblib' else original(path)
        with patch.object(Path, 'is_file', fake):
            at = self.new_app()
            self.assertTrue(any('final_model.joblib' in e.value for e in at.error))
            self.assertFalse(at.exception)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'all_passed': result.wasSuccessful(), 'browser_visual_qa': 'not performed; browser skill unavailable',
              'ui_test_method':'Streamlit AppTest; CSV upload object injected to exercise handler',
              'live_cloud_deployment':'not performed; requires user GitHub/Streamlit account'}
    (ROOT/'results/web_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
