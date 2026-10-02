"""Offline unit/integration tests; camera transport and YOLO are simulated."""

import contextlib
import io
import os
import runpy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import joblib
import numpy as np
import pandas as pd
from PyQt5.QtCore import Q_ARG, QMetaObject, Qt
from PyQt5.QtWidgets import QApplication

from core.forecast_worker import ForecastWorker
from core.traffic_features import (
    CONGESTION_THRESHOLD, DISPLAY_WINDOW, FEATURE_NAMES, FORECAST_HORIZON, HISTORY_POINTS,
    HISTORY_WINDOW, MINUTE_COLUMNS, MODEL_PATH, MODEL_VERSION, RESAMPLE_INTERVAL,
    REFERENCE_INTERVAL_MINUTES, SMOOTHING_WINDOW, TARGET_COLUMN, build_daily_profile, create_features,
    create_targets, read_minutes, read_samples, reference_minutes, resample_minutes,
    smooth_forecast, smooth_observed, supervised_data,
)
from core.traffic_forecast import TrafficForecaster, predict_values, stable_crossing, validate_bundle
from core.vision_worker import CameraWorker
from ui.tabs.dashboard_tab import DashboardTab
from utils.app_state import AppState, app_state

# Keep one QApplication alive for the suite; destroying it deletes Fluent's QConfig.
APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]
TRAINING = runpy.run_path(str(ROOT / "training_model.Py"))


class FakeModel:
    """Deterministic schema-compatible test double. No estimator is fitted."""
    feature_names_in_ = np.array(FEATURE_NAMES)
    estimators_ = [None] * FORECAST_HORIZON

    def predict(self, features):
        return np.repeat(features[TARGET_COLUMN].to_numpy()[:, None], FORECAST_HORIZON, axis=1)


def fake_bundle(profile):
    return {
        "model": FakeModel(), "feature_names": FEATURE_NAMES,
        "forecast_horizon": FORECAST_HORIZON, "history_points": HISTORY_POINTS,
        "history_window": HISTORY_WINDOW, "resample_interval": RESAMPLE_INTERVAL,
        "model_version": MODEL_VERSION, "target_column": TARGET_COLUMN,
        "target_mode": "absolute", "smoothing_window": SMOOTHING_WINDOW,
        "congestion_threshold": CONGESTION_THRESHOLD, "metrics": {},
        "reference_interval_minutes": REFERENCE_INTERVAL_MINUTES, "daily_profile": profile,
    }


def wait_for(app, condition, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minutes = read_minutes(ROOT / "data/data1.csv")

    def test_minute_resampling_and_missing(self):
        rows = pd.DataFrame({
            "timestamp": ["2026-09-21 10:00:05", "2026-09-21 10:00:35", "2026-09-21 10:02:05"],
            "car": [2, 4, 1], "motorbike": [1, 1, 0], "bus": [0, 0, 0],
            "truck": [0, 0, 0], "total": [3, 5, 1], "WTI": [9, 17, 4],
        })
        minutes = resample_minutes(rows)
        self.assertEqual(minutes.index[0], pd.Timestamp("2026-09-21 10:01"))
        self.assertEqual(minutes.total_mean.iloc[0], 4)
        self.assertEqual(minutes.total_std.iloc[0], 1)
        self.assertEqual(minutes.total_max.iloc[0], 5)
        self.assertEqual(minutes.samples_per_minute.iloc[0], 2)
        self.assertEqual(minutes.WTI_mean.iloc[0], 13)
        self.assertEqual(minutes.WTI_std.iloc[0], 4)
        self.assertEqual(minutes.WTI_max.iloc[0], 17)
        self.assertTrue(pd.isna(minutes.total_mean.iloc[1]))
        self.assertEqual(minutes.samples_per_minute.iloc[1], 0)
        self.assertEqual(minutes.total_std.iloc[2], 0)

    def test_features_do_not_use_future(self):
        origin = self.minutes.index[100]
        before = create_features(self.minutes).loc[:origin]
        changed = self.minutes.copy()
        changed.loc[changed.index > origin, "total_mean"] = 10000
        changed.loc[changed.index > origin, "WTI_mean"] = 20000
        pd.testing.assert_frame_equal(before, create_features(changed).loc[:origin])
        pd.testing.assert_series_equal(before.iloc[-1], create_features(self.minutes.loc[:origin]).iloc[-1])

    def test_feature_order_and_lag60(self):
        features = create_features(self.minutes)
        self.assertEqual(list(features.columns), FEATURE_NAMES)
        self.assertEqual(features.total_lag_60.iloc[100], self.minutes.total_mean.iloc[40])
        self.assertEqual(features.WTI_lag_60.iloc[100], self.minutes.WTI_mean.iloc[40])
        self.assertFalse(any("WTI_norm" in name for name in FEATURE_NAMES))
        self.assertTrue(features.iloc[:60].isna().all().all())
        self.assertFalse(features.iloc[60].isna().any())

    def test_targets_are_exact_future_minutes(self):
        targets = create_targets(self.minutes)
        self.assertEqual(targets.shape[1], FORECAST_HORIZON)
        self.assertEqual(targets.iloc[100, 0], self.minutes.WTI_mean.iloc[101])
        self.assertEqual(targets.iloc[100, -1], self.minutes.WTI_mean.iloc[160])
        self.assertEqual(list(targets.columns), [f"WTI_t+{step}" for step in range(1, 61)])
        self.assertTrue(targets.iloc[-1].isna().all())

    def test_missing_minute_invalidates_history_and_targets(self):
        minutes = self.minutes.iloc[:160].drop(self.minutes.index[80])
        features = create_features(minutes)
        self.assertTrue(features.iloc[80:141].isna().all().all())
        self.assertFalse(features.iloc[141].isna().any())
        self.assertTrue(pd.isna(create_targets(minutes).iloc[79, 0]))
        x, y = supervised_data(minutes)
        self.assertTrue(x.notna().all().all() and y.notna().all().all())

    def test_schema_rejects_invalid_counts(self):
        rows = read_samples(ROOT / "data/data1.csv").iloc[:2].copy()
        rows.loc[rows.index[0], "total"] = -1
        with self.assertRaises(ValueError):
            resample_minutes(rows)

    def test_daily_reference_is_training_only_and_handles_midnight(self):
        profile = build_daily_profile(self.minutes)
        self.assertEqual(profile.shape, (96, len(MINUTE_COLUMNS)))
        times = pd.date_range("2026-10-01 23:59", periods=2, freq="1min")
        mapped = reference_minutes(profile, times)
        np.testing.assert_allclose(mapped.iloc[0], profile.iloc[95])
        np.testing.assert_allclose(mapped.iloc[1], profile.iloc[0])
        self.assertEqual(list(mapped.index), list(times))

    def test_viewer_detection_is_off_by_default(self):
        self.assertFalse(AppState().show_yolo_frame)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "smoke.joblib"
        profile = build_daily_profile(read_minutes(ROOT / "data/data1.csv"))
        joblib.dump(fake_bundle(profile), cls.path)
        cls.minutes = read_minutes(ROOT / "data/data2.csv")
        cls.x, cls.y = supervised_data(cls.minutes)
        cls.history = cls.minutes.loc[:cls.x.index[100]].tail(HISTORY_POINTS)
        cls.forecaster = TrafficForecaster(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_bundle_serialization_without_training(self):
        bundle = validate_bundle(joblib.load(self.path))
        self.assertEqual(bundle["feature_names"], list(self.x.columns))
        self.assertEqual(bundle["target_column"], "WTI_mean")
        pd.testing.assert_frame_equal(bundle["daily_profile"], build_daily_profile(read_minutes(ROOT / "data/data1.csv")))

    def test_inference_same_features_as_train(self):
        pd.testing.assert_frame_equal(create_features(self.history).iloc[[-1]], self.x.loc[[self.history.index[-1]]], check_freq=False)

    def test_60_predictions_and_timestamp_alignment(self):
        result = self.forecaster.predict(self.history)
        self.assertTrue(result.is_ready, result.status_message)
        self.assertEqual(len(result.forecast_values), 60)
        self.assertEqual(len(result.forecast_times), 60)
        self.assertEqual(result.forecast_times[0], self.history.index[-1] + pd.Timedelta(minutes=1))
        self.assertEqual(result.forecast_times[-1], self.history.index[-1] + pd.Timedelta(minutes=60))
        self.assertTrue(np.isfinite(result.forecast_values).all())

    def test_three_of_five_and_eta_start(self):
        self.assertEqual(stable_crossing([0, 0, 101, 101, 101]), 1)
        self.assertEqual(stable_crossing([0, 0, 0, 0, 0, 101, 101, 101]), 4)
        self.assertIsNone(stable_crossing([101, 0, 0, 0, 0, 101, 0, 0, 0, 0]))
        self.assertIsNone(stable_crossing([100] * 60))
        self.assertIsNone(stable_crossing([101, 101, 101]))

    def test_eta_crossing_time_and_alert_levels(self):
        for values in ([110] * 60, [10] * 10 + [110] * 50, [10] * 60):
            displayed = smooth_forecast(values, self.history.WTI_mean.tail(2))
            eta = stable_crossing(displayed)
            level = "NORMAL" if eta is None else ("WARNING" if eta <= 5 else "WATCH")
            with patch.object(self.forecaster.bundle["model"], "predict", return_value=np.array([values])):
                result = self.forecaster.predict(self.history)
            self.assertEqual(result.eta_minutes, eta)
            self.assertEqual(result.alert_level, level)
            np.testing.assert_allclose(result.forecast_values, displayed)
            self.assertEqual(result.crossing_time, None if eta is None else self.history.index[-1] + pd.Timedelta(minutes=eta))

    def test_insufficient_history_and_gap(self):
        self.assertEqual(self.forecaster.predict(self.history.iloc[1:]).alert_level, "INSUFFICIENT_DATA")
        self.assertEqual(self.forecaster.predict(self.history.drop(self.history.index[30])).alert_level, "INSUFFICIENT_DATA")

    def test_missing_model(self):
        result = TrafficForecaster(self.path.with_name("missing.joblib")).predict(self.history)
        self.assertEqual(result.alert_level, "MODEL_ERROR")
        self.assertFalse(result.is_ready)

    def test_bad_bundle_and_output(self):
        bundle = joblib.load(self.path)
        bundle["feature_names"] = list(reversed(FEATURE_NAMES))
        with self.assertRaises(ValueError):
            validate_bundle(bundle)
        with patch.object(self.forecaster.bundle["model"], "predict", return_value=np.zeros((1, 59))):
            self.assertEqual(self.forecaster.predict(self.history).alert_level, "MODEL_ERROR")

    def test_load_only_once(self):
        with patch("core.traffic_forecast.joblib.load", wraps=joblib.load) as loader:
            forecaster = TrafficForecaster(self.path)
            forecaster.predict(self.history)
            forecaster.predict(self.history)
            self.assertEqual(loader.call_count, 1)

    def test_existing_model_is_preserved(self):
        before = self.path.read_bytes()
        candidate = TRAINING["choose_output_path"](self.path)
        self.assertNotEqual(candidate, self.path)
        self.assertEqual(before, self.path.read_bytes())

    def test_architecture_and_describe_never_fit(self):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        with patch.object(MultiOutputRegressor, "fit", side_effect=AssertionError("Training forbidden")), patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("Training forbidden")):
            model = TRAINING["build_model"]()
            self.assertIsInstance(model.estimator, HistGradientBoostingRegressor)
            self.assertFalse(model.estimator.early_stopping)
            with patch.object(sys, "argv", ["training_model.Py", "--describe"]), contextlib.redirect_stdout(io.StringIO()):
                TRAINING["main"]()

    def test_time_validation_gap_and_delta_targets(self):
        fit, val = TRAINING["chronological_split"](self.x)
        self.assertLess(self.x.index[fit][-1] + pd.Timedelta(minutes=60), self.x.index[val][0])
        delta = TRAINING["training_targets"](self.y, self.x, "delta")
        np.testing.assert_allclose(delta.to_numpy() + self.x.WTI_mean.to_numpy()[:, None], self.y.to_numpy())
        bundle = dict(self.forecaster.bundle, target_mode="delta")
        np.testing.assert_allclose(predict_values(bundle, self.x.iloc[:1]), self.x.WTI_mean.iloc[0] * 2)

    def test_three_minute_smoothing_and_no_reference_blending(self):
        np.testing.assert_allclose(smooth_forecast([3, 9, 6], [0, 0]), [1, 4, 6])
        history = self.history.copy()
        history["is_reference"] = True
        history.loc[history.index[-3:], "is_reference"] = False
        history.loc[history.index[-3:], "WTI_mean"] = [8, 2, 4]
        np.testing.assert_allclose(smooth_observed(history).iloc[-3:], [8, 5, 14 / 3])
        history.loc[history.index[-2], "WTI_mean"] = np.nan
        self.assertEqual(smooth_observed(history).iloc[-1], 4)

    def test_worker_finalizes_minutes_and_rejects_stale_camera(self):
        worker = ForecastWorker(self.path)
        worker.forecaster = self.forecaster
        worker.camera_name = "Camera 1"
        origin = self.history.index[-1]
        rows = read_samples(ROOT / "data/data2.csv")
        worker.samples = rows.loc[(rows.timestamp >= self.history.index[0] - pd.Timedelta(minutes=1)) & (rows.timestamp < origin)].to_dict("records")
        updates = []
        worker.updated.connect(lambda history, result, camera_name: updates.append((history, result)))
        worker.refresh(origin)
        self.assertTrue(updates[-1][1].is_ready)
        self.assertEqual(updates[-1][0].index[-1], origin)
        worker.refresh(origin + pd.Timedelta(seconds=30))
        self.assertEqual(len(updates), 1)
        worker.refresh(origin + pd.Timedelta(minutes=1))
        self.assertEqual(updates[-1][1].alert_level, "INSUFFICIENT_DATA")
        before = len(worker.samples)
        worker.add_sample({"camera_name": "Camera 2", "timestamp": origin})
        self.assertEqual(len(worker.samples), before)

    def test_reference_warm_start_replaced_by_realtime_and_outages_not_filled(self):
        worker = ForecastWorker(self.path)
        worker.forecaster = self.forecaster
        worker.daily_profile = self.forecaster.bundle["daily_profile"]
        origin = pd.Timestamp("2026-10-01 09:15")
        worker.session_start_minute = origin
        updates = []
        worker.updated.connect(lambda history, result, camera: updates.append((history, result)))
        worker.refresh(origin)
        history, result = updates[-1]
        self.assertTrue(result.is_ready)
        self.assertTrue(result.is_provisional)
        self.assertEqual(result.reference_minutes, HISTORY_POINTS)
        self.assertIn("tham chiếu", result.status_message)
        self.assertEqual(len(history.attrs["daily_reference"]), 241)
        worker.samples = [{
            "timestamp": origin + pd.Timedelta(minutes=step, seconds=second),
            "car": 2, "motorbike": 3, "bus": 0, "truck": 1, "total": 6, "WTI": 17,
        } for step in range(HISTORY_POINTS) for second in (10, 25, 40)]
        worker.refresh(origin + pd.Timedelta(minutes=1))
        self.assertEqual(updates[-1][1].reference_minutes, HISTORY_POINTS - 1)
        self.assertEqual(updates[-1][0].total_mean.iloc[-1], 6)
        self.assertLess(abs(updates[-1][1].forecast_values[0] - 17), 5)
        worker.refresh(origin + pd.Timedelta(minutes=HISTORY_POINTS))
        self.assertTrue(updates[-1][1].is_ready)
        self.assertEqual(updates[-1][1].reference_minutes, 0)
        self.assertFalse(updates[-1][1].is_provisional)
        worker.refresh(origin + pd.Timedelta(minutes=HISTORY_POINTS + 1))
        self.assertEqual(updates[-1][1].alert_level, "INSUFFICIENT_DATA")
        self.assertTrue(pd.isna(updates[-1][0].total_mean.iloc[-1]))

    def test_widget_and_dashboard_queued_replay(self):
        app = QApplication.instance() or QApplication([])
        dashboard = DashboardTab(model_path=self.path, store_path=None)
        dashboard.resize(1100, 700)
        dashboard.show()
        try:
            self.assertTrue(wait_for(app, lambda: dashboard.forecast_worker.forecaster is not None))
            self.assertTrue(wait_for(app, lambda: hasattr(dashboard.forecast_worker, "timer")))
            QMetaObject.invokeMethod(dashboard.forecast_worker.timer, "stop", Qt.BlockingQueuedConnection)
            origin = self.history.index[-1]
            rows = read_samples(ROOT / "data/data2.csv")
            rows = rows.loc[(rows.timestamp >= self.history.index[0] - pd.Timedelta(minutes=1)) & (rows.timestamp < origin)]
            # Use the same queued API as camera counts; historical clock is explicit.
            for row in rows.to_dict("records"):
                dashboard.receive_sample(dict(row, camera_name="Camera 1"))
            QMetaObject.invokeMethod(dashboard.forecast_worker, "refresh", Qt.QueuedConnection, Q_ARG(object, origin))
            self.assertTrue(wait_for(app, lambda: dashboard.chart.current_time == origin and len(dashboard.chart.forecast_line.get_ydata()) == 61))
            self.assertEqual(len(dashboard.chart.actual_line.get_ydata()), DISPLAY_WINDOW + 1)
            self.assertNotEqual(dashboard.forecast_worker.thread(), app.thread())
            figure_id, artist_ids = id(dashboard.chart.figure), [id(line) for line in dashboard.chart.ax.lines]
            result = self.forecaster.predict(self.history)
            dashboard.update_forecast(self.history, result)
            dashboard.update_forecast(self.history, result)
            self.assertEqual(id(dashboard.chart.figure), figure_id)
            self.assertEqual([id(line) for line in dashboard.chart.ax.lines], artist_ids)
            dashboard.chart.canvas.draw()
            self.assertTrue(dashboard.grab().save(str(Path(self.directory.name) / "dashboard.png")))
            before = dashboard.chart.forecast_line.get_ydata().copy()
            dashboard._on_worker_update(self.history, result, "Camera 2")
            np.testing.assert_array_equal(dashboard.chart.forecast_line.get_ydata(), before)
            dashboard.reference_button.click()
            self.assertTrue(dashboard.chart.reference_line.get_visible())
            dashboard.reference_button.click()
            self.assertFalse(dashboard.chart.reference_line.get_visible())
            left, right = dashboard.chart.ax.get_xlim()
            import matplotlib.dates as mdates
            self.assertAlmostEqual((left + right) / 2, mdates.date2num(dashboard.chart.current_time))
            self.assertAlmostEqual(right - left, 240 / (24 * 60))
        finally:
            dashboard.stop()
            dashboard.close()

    def test_ui_missing_model_does_not_crash(self):
        app = QApplication.instance() or QApplication([])
        dashboard = DashboardTab(model_path=self.path.with_name("missing.joblib"), store_path=None)
        try:
            self.assertTrue(wait_for(app, lambda: "Chưa có model WTI" in dashboard.status_label.text()))
            self.assertEqual(len(dashboard.chart.forecast_line.get_ydata()), 61)
            self.assertTrue(dashboard.last_result.is_provisional)
        finally:
            dashboard.stop()
            dashboard.close()

    def test_camera_worker_emits_nine_value_detection(self):
        app = QApplication.instance() or QApplication([])
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        capture = types.ModuleType("core.Get_frame")
        capture.get_traffic_image = lambda name: frame
        detector = types.ModuleType("core.Yolo_detect")
        detector.image_processing = lambda image: (image, 1, 2, 3, 0, 1, 6, 17.0, 20.0)

        class OneFrameWorker(CameraWorker):
            def msleep(self, milliseconds):
                self.requestInterruption()

        worker = OneFrameWorker()
        counts, frames = [], []
        worker.counts_ready.connect(counts.append)
        worker.frame_ready.connect(frames.append)
        with patch.dict(sys.modules, {"core.Get_frame": capture, "core.Yolo_detect": detector}):
            worker.start()
            self.assertTrue(wait_for(app, lambda: not worker.isRunning()))
            app.processEvents()
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0]["total"], 6)
        self.assertEqual(counts[0]["WTI"], 17)
        self.assertEqual(len(frames), 1)

    def test_viewer_camera_and_toggle_do_not_change_forecast_source(self):
        app = APP
        capture = types.ModuleType("core.Get_frame")
        detector = types.ModuleType("core.Yolo_detect")

        class OneFrameWorker(CameraWorker):
            def msleep(self, milliseconds):
                self.requestInterruption()

        for viewer in ("Camera 1", "Camera 2"):
            for checked in (False, True):
                requested, detected, counts, frames = [], [], [], []

                def get_frame(name):
                    requested.append(name)
                    return np.full((30, 30, 3), 10 if name == "Camera 1" else 20, dtype=np.uint8)

                def detect(image):
                    detected.append(int(image[0, 0, 0]))
                    return image + 60, 1, 2, 3, 0, 1, 6, 17.0, 20.0

                capture.get_traffic_image, detector.image_processing = get_frame, detect
                worker = OneFrameWorker()
                worker.counts_ready.connect(counts.append)
                worker.frame_ready.connect(frames.append)
                with patch.object(app_state, "selected_camera_name", viewer), patch.object(app_state, "show_yolo_frame", checked), patch.dict(sys.modules, {"core.Get_frame": capture, "core.Yolo_detect": detector}):
                    worker.start()
                    self.assertTrue(wait_for(app, lambda: not worker.isRunning()))
                    app.processEvents()
                self.assertEqual(counts[0]["camera_name"], "Camera 1")
                self.assertEqual(requested[0], "Camera 1")
                self.assertEqual(detected, [10, 20] if checked and viewer == "Camera 2" else [10])
                expected = (10 if viewer == "Camera 1" else 20) + (60 if checked else 0)
                self.assertEqual(frames[0].pixelColor(0, 0).blue(), expected)


if __name__ == "__main__":
    unittest.main()
