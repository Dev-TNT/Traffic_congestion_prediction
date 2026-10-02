"""Regression tests for adaptive startup, persistence, quality and honest UI."""

from pathlib import Path
import tempfile
import unittest
import runpy
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_traffic_forecast import APP, FakeModel, fake_bundle, wait_for
from core.traffic_features import MINUTE_COLUMNS, reference_minutes
from core.traffic_forecast import TrafficForecaster, provisional_values
from core.forecast_worker import ForecastWorker
from core.traffic_store import TrafficStore
from ui.tabs.dashboard_tab import DashboardTab


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.profile = pd.DataFrame(50.0, index=range(96), columns=MINUTE_COLUMNS)
        self.profile.samples_per_minute = 5
        self.times = pd.date_range("2026-10-02 12:00", periods=61, freq="min")
        self.history = reference_minutes(self.profile, self.times)
        self.history["is_reference"] = False
        self.future = pd.date_range(self.times[-1] + pd.Timedelta(minutes=1), periods=60, freq="min")
        self.forecaster = TrafficForecaster(Path(self.directory.name) / "missing.joblib")
        self.forecaster.bundle = fake_bundle(self.profile)
        self.forecaster.error = None

    def tearDown(self):
        self.directory.cleanup()

    def sample(self, timestamp):
        return dict(timestamp=timestamp, car=2, motorbike=3, bus=0, truck=1, total=6, WTI=17)

    def test_earlier_realtime_changes_forecast_even_with_same_last_three(self):
        first = self.history.copy()
        second = self.history.copy()
        second.loc[second.index[-30:-3], "WTI_mean"] = 20
        a = provisional_values(first, self.profile, self.future)
        b = provisional_values(second, self.profile, self.future)
        self.assertGreater(np.max(np.abs(a - b)), 1)

    def test_correction_decays_and_future_observations_are_not_required(self):
        self.history.WTI_mean = 80
        values = provisional_values(self.history, self.profile, self.future)
        self.assertGreater(values[0], values[-1])
        self.assertLess(abs(values[-1] - 50), 2)
        self.assertTrue(np.isfinite(values).all())

    def test_outage_is_not_bridged_by_adaptive_trend(self):
        self.history.loc[self.times[-4], "WTI_mean"] = np.nan
        a = provisional_values(self.history, self.profile, self.future)
        self.history.loc[self.times[:-4], "WTI_mean"] = 500
        np.testing.assert_allclose(a, provisional_values(self.history, self.profile, self.future))

    def test_old_gap_allows_provisional_but_current_gap_stops_prediction(self):
        self.history.loc[self.times[10], "WTI_mean"] = np.nan
        result = self.forecaster.predict(self.history)
        self.assertTrue(result.is_ready)
        self.assertTrue(result.is_provisional)
        self.assertEqual(result.alert_level, "PROVISIONAL")
        self.history.loc[self.times[-1], "WTI_mean"] = np.nan
        self.assertFalse(self.forecaster.predict(self.history).is_ready)

    def test_sparse_minutes_do_not_enter_model(self):
        self.history.loc[self.times[-1], "samples_per_minute"] = 1
        with patch.object(FakeModel, "predict", side_effect=AssertionError("Sparse history reached model")):
            result = self.forecaster.predict(self.history)
        self.assertTrue(result.is_provisional)
        self.assertIn("1 phút dưới 3 mẫu", result.quality_message)

    def test_ongoing_exceedance_has_zero_eta(self):
        self.history.loc[self.times[-5:], "WTI_mean"] = 110
        result = self.forecaster.predict(self.history)
        self.assertEqual(result.eta_minutes, 0)
        self.assertEqual(result.crossing_time, self.times[-1])
        self.assertEqual(result.alert_level, "WARNING")

    def test_store_reopens_deduplicates_and_preserves_forecast_versions(self):
        store = TrafficStore(self.path)
        sample = self.sample(self.times[-1])
        store.save_sample(sample)
        store.save_sample(sample)
        result = self.forecaster.predict(self.history)
        store.save_forecast(self.times[-1], result, "test-model")
        store.close()
        restored = TrafficStore(self.path)
        try:
            rows = restored.samples_since(self.times[0])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["WTI"], 17)
            self.assertEqual(restored.connection.execute("SELECT count(*) FROM forecasts").fetchone()[0], 1)
        finally:
            restored.close()

    def test_worker_restores_and_does_not_fill_downtime_with_reference(self):
        now = pd.Timestamp.now().floor("min")
        store = TrafficStore(self.path)
        store.save_sample(self.sample(now - pd.Timedelta(minutes=3, seconds=10)))
        store.close()
        worker = ForecastWorker(store_path=self.path)
        output = []
        worker.updated.connect(lambda h, r, c: output.append((h, r)))
        with patch("core.forecast_worker.TrafficForecaster", return_value=self.forecaster):
            worker.start()
        try:
            self.assertEqual(len(worker.samples), 1)
            history, result = output[-1]
            self.assertTrue(pd.isna(history.WTI_mean.iloc[-1]))
            self.assertFalse(history.is_reference.iloc[-1])
            self.assertFalse(result.is_ready)
        finally:
            worker.stop()

    def test_late_sample_refreshes_completed_minute(self):
        now = pd.Timestamp.now().floor("min")
        worker = ForecastWorker(store_path=None)
        worker.forecaster, worker.daily_profile = self.forecaster, self.profile
        worker.session_start_minute = now - pd.Timedelta(minutes=1)
        output = []
        worker.updated.connect(lambda h, r, c: output.append((h, r)))
        worker.refresh(now)
        self.assertFalse(output[-1][1].is_ready)
        worker.add_sample(dict(self.sample(now - pd.Timedelta(seconds=10)), camera_name="Camera 1"))
        self.assertTrue(output[-1][1].is_ready)
        self.assertEqual(output[-1][0].WTI_mean.iloc[-1], 17)

    def test_provisional_dashboard_never_promises_no_congestion(self):
        dashboard = DashboardTab(model_path=Path(self.directory.name) / "missing.joblib", store_path=None)
        try:
            self.assertTrue(wait_for(APP, lambda: hasattr(dashboard, "last_result")))
            self.history.is_reference = True
            result = self.forecaster.predict(self.history)
            dashboard.update_forecast(self.history, result)
            self.assertEqual(result.alert_level, "PROVISIONAL")
            self.assertEqual(dashboard.summary_labels["eta"].text(), "Chưa thấy vượt")
        finally:
            dashboard.stop()
            dashboard.close()

    def test_saved_forecast_scores_only_complete_future_and_first_revision(self):
        origin = pd.Timestamp.now().floor("min") - pd.Timedelta(days=1)
        store = TrafficStore(self.path)
        try:
            for minute in range(60):
                for seconds in (5, 20, 40):
                    store.save_sample(self.sample(origin + pd.Timedelta(minutes=minute, seconds=seconds)))
            result = self.forecaster.predict(self.history)
            result.forecast_values = np.full(60, 17.0)
            result.forecast_times = pd.date_range(origin + pd.Timedelta(minutes=1), periods=60, freq="min")
            # Real issuance time precedes the first future minute's completion.
            store.save_forecast(origin, result, "test")
            with store.connection:
                store.connection.execute("UPDATE forecasts SET issued_at = ?", ((origin + pd.Timedelta(seconds=1)).isoformat(),))
            result.forecast_values = np.full(60, 500.0)
            store.save_forecast(origin, result, "test")
        finally:
            store.close()
        evaluate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_live_forecasts.py"))["evaluate_store"]
        report = evaluate(self.path)
        metric = next(iter(report["groups"].values()))
        self.assertEqual(metric["mean_mae"], 0)
        self.assertEqual(metric["threshold_events"]["evaluated_origins"], 1)

    def test_collector_flushes_on_detection_error_and_preserves_existing_file(self):
        capture = types.ModuleType("core.Get_frame")
        capture.get_traffic_image = lambda name: np.zeros((10, 10, 3), dtype=np.uint8)
        capture.cam_id = {"Camera 1": "test"}
        detector = types.ModuleType("core.Yolo_detect")
        detector.image_processing = lambda frame: (frame, 0, 2, 3, 0, 1, 6, 17, 20)
        with patch.dict(sys.modules, {"core.Get_frame": capture, "core.Yolo_detect": detector}):
            collector = runpy.run_path(str(Path(__file__).resolve().parents[1] / "get_data.py"))
        path = Path(self.directory.name) / "capture.csv"
        frames = [np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10, 3), dtype=np.uint8)]
        with patch.dict(collector["collect_data"].__globals__, {
            "get_traffic_image": lambda name: frames.pop(0),
            "image_processing": unittest.mock.Mock(side_effect=[detector.image_processing(frames[0]), RuntimeError("detector stopped")]),
        }), patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1), patch("cv2.destroyAllWindows"):
            with self.assertRaises(RuntimeError):
                collector["collect_data"](output_path=str(path))
        self.assertEqual(len(pd.read_csv(path)), 1)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            collector["collect_data"](output_path=str(path))
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
