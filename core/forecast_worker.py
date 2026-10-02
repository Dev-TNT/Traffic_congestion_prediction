"""Qt worker owns minute history and the single loaded forecast model."""

import pandas as pd
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from core.traffic_features import (
    FORECAST_CAMERA, DISPLAY_WINDOW, MINUTE_COLUMNS, MODEL_PATH,
    RESAMPLE_INTERVAL, build_daily_profile, read_minutes, reference_minutes, resample_minutes, validate_samples,
)
from core.traffic_forecast import ForecastResult, TrafficForecaster
from core.traffic_store import STORE_PATH, TrafficStore


class ForecastWorker(QObject):
    updated = pyqtSignal(object, object, str)

    def __init__(self, model_path=MODEL_PATH, store_path=STORE_PATH):
        super().__init__()
        self.model_path = model_path
        self.samples = []
        self.camera_name = FORECAST_CAMERA
        self.daily_profile = None
        self.session_start_minute = None
        self.last_minute = None
        self.forecaster = None
        self.store_path = store_path
        self.store = None
        self.storage_error = ""

    @pyqtSlot()
    def start(self):
        self.forecaster = TrafficForecaster(self.model_path)
        self.session_start_minute = pd.Timestamp.now().floor(RESAMPLE_INTERVAL)
        if self.store_path is not None:
            try:
                self.store = TrafficStore(self.store_path)
                self.samples = self.store.samples_since(self.session_start_minute - pd.Timedelta(minutes=DISPLAY_WINDOW + 1))
                if self.samples:
                    # Never turn a recorded gap or downtime into reference history.
                    self.session_start_minute = pd.Timestamp(self.samples[0]["timestamp"]).floor(RESAMPLE_INTERVAL)
            except Exception as error:
                self.storage_error = f"Không lưu/khôi phục được lịch sử: {error}"
        if self.forecaster.bundle is not None:
            self.daily_profile = self.forecaster.bundle["daily_profile"]
        else:
            # Reference display remains available even before the first model train.
            try:
                self.daily_profile = build_daily_profile(read_minutes(MODEL_PATH.parents[1] / "data/data1.csv"))
            except (OSError, ValueError):
                self.daily_profile = None
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # check boundary; predict only when a minute closes
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    @pyqtSlot(object)
    def add_sample(self, sample):
        if sample["camera_name"] != self.camera_name:
            return  # viewer camera counts can never enter Camera 1's forecast history
        try:
            validate_samples(pd.DataFrame([sample]))
        except (ValueError, TypeError):
            return
        row = {key: value for key, value in sample.items() if key != "camera_name"}
        if any(pd.Timestamp(existing["timestamp"]) == pd.Timestamp(row["timestamp"]) for existing in self.samples):
            return
        self.samples.append(row)
        if self.store is not None:
            try:
                self.store.save_sample(row)
            except Exception as error:
                self.storage_error = f"Không lưu được lịch sử: {error}"
        # A completed minute can receive a delayed detection result. Recompute it.
        if self.last_minute is not None and self.last_minute - pd.Timedelta(minutes=1) <= pd.Timestamp(row["timestamp"]) < self.last_minute:
            self.last_minute = None
        self.refresh()

    @pyqtSlot()
    def stop(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.store is not None:
            self.store.close()
            self.store = None

    @pyqtSlot()
    @pyqtSlot(object)
    def refresh(self, now=None):
        if self.forecaster is None:
            return
        now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
        minute = now.floor(RESAMPLE_INTERVAL)
        if minute == self.last_minute:
            return
        self.last_minute = minute
        cutoff = minute - pd.Timedelta(minutes=DISPLAY_WINDOW + 1)
        self.samples = [sample for sample in self.samples if pd.Timestamp(sample["timestamp"]) >= cutoff]
        try:
            index = pd.date_range(minute - pd.Timedelta(minutes=DISPLAY_WINDOW), minute, freq=RESAMPLE_INTERVAL)
            history = pd.DataFrame(float("nan"), index=index, columns=MINUTE_COLUMNS)
            history["samples_per_minute"] = 0.0
            history["is_reference"] = False
            start = self.session_start_minute if self.session_start_minute is not None else index[0] - pd.Timedelta(minutes=1)
            if self.daily_profile is not None:
                # Fill only pre-start history; never mask a post-start camera outage.
                references = reference_minutes(self.daily_profile, index)
                history.loc[index <= start, list(MINUTE_COLUMNS)] = references.loc[index <= start]
                history.loc[index <= start, "is_reference"] = True
                comparison_times = pd.date_range(minute - pd.Timedelta(minutes=DISPLAY_WINDOW), minute + pd.Timedelta(minutes=DISPLAY_WINDOW), freq=RESAMPLE_INTERVAL)
                history.attrs["daily_reference"] = reference_minutes(self.daily_profile, comparison_times)
            if self.samples:
                observed = resample_minutes(pd.DataFrame(self.samples)).reindex(index)
                valid = observed.samples_per_minute > 0
                history.loc[valid, list(MINUTE_COLUMNS)] = observed.loc[valid, list(MINUTE_COLUMNS)]
                history.loc[valid, "is_reference"] = False
            history.attrs["current_time"] = minute
            if self.samples:
                latest = max(pd.Timestamp(sample["timestamp"]) for sample in self.samples)
                history.attrs["last_received"] = latest
                history.attrs["sample_age_seconds"] = max(0, (now - latest).total_seconds())
            result = self.forecaster.predict(history, self.daily_profile)
            if self.store is not None:
                try:
                    bundle = self.forecaster.bundle or {}
                    self.store.save_forecast(minute, result, bundle.get("created_at_utc", "unavailable"))
                except Exception as error:
                    self.storage_error = f"Không lưu được dự báo: {error}"
            if self.storage_error:
                result.status_message += " · " + self.storage_error
        except Exception as error:
            history = pd.DataFrame()
            result = ForecastResult(alert_level="MODEL_ERROR", status_message=f"Lỗi dữ liệu realtime: {error}")
        self.updated.emit(history, result, self.camera_name or "")
