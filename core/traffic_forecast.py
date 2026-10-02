"""UI-independent model loading, prediction and stable threshold ETA."""

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import warnings
from threadpoolctl import threadpool_limits

from core.traffic_features import (
    CONGESTION_THRESHOLD, FEATURE_NAMES, FORECAST_HORIZON, HISTORY_POINTS,
    HISTORY_WINDOW, MODEL_PATH, MODEL_VERSION, RESAMPLE_INTERVAL, create_features,
    MINUTE_COLUMNS, REFERENCE_INTERVAL_MINUTES, TARGET_COLUMN, SMOOTHING_WINDOW, MIN_SAMPLES_PER_MINUTE,
    regular_minutes, reference_minutes, smooth_forecast,
)


ALERT_WINDOW = 5
ALERT_MINUTES = 3


@dataclass
class ForecastResult:
    forecast_times: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    forecast_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    threshold: float = CONGESTION_THRESHOLD
    eta_minutes: int | None = None
    crossing_time: pd.Timestamp | None = None
    alert_level: str = "INSUFFICIENT_DATA"
    is_ready: bool = False
    status_message: str = "Chưa đủ dữ liệu để dự báo."
    reference_minutes: int = 0
    is_provisional: bool = False
    raw_forecast_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    real_minutes: int = 0
    quality_message: str = ""
    model_error: str | None = None


def stable_crossing(values, threshold=CONGESTION_THRESHOLD) -> int | None:
    """Return 1-based start of first COMPLETE five-minute window with 3 exceedances.

    The start itself need not exceed the threshold. Equality is not exceedance.
    A horizon's incomplete final window cannot trigger an alert.
    """
    values = np.asarray(values, dtype=float)
    for start in range(len(values) - ALERT_WINDOW + 1):
        window = values[start:start + ALERT_WINDOW]
        if np.isfinite(window).all() and np.count_nonzero(window > threshold) >= ALERT_MINUTES:
            return start + 1
    return None


def validate_bundle(bundle):
    expected = {
        "feature_names": FEATURE_NAMES, "forecast_horizon": FORECAST_HORIZON,
        "resample_interval": RESAMPLE_INTERVAL, "history_window": HISTORY_WINDOW,
        "history_points": HISTORY_POINTS, "model_version": MODEL_VERSION,
        "reference_interval_minutes": REFERENCE_INTERVAL_MINUTES,
        "target_column": TARGET_COLUMN, "smoothing_window": SMOOTHING_WINDOW,
    }
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise ValueError(f"Incompatible model metadata: {key}")
    if "metrics" not in bundle or "congestion_threshold" not in bundle:
        raise ValueError("Model metadata is incomplete")
    if bundle["congestion_threshold"] != CONGESTION_THRESHOLD:
        raise ValueError("Threshold changed; retrain the bundle with the current config")
    model = bundle.get("model")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Model has no predict method")
    if list(getattr(model, "feature_names_in_", [])) != FEATURE_NAMES:
        raise ValueError("Estimator feature order differs from bundle")
    if len(getattr(model, "estimators_", [])) != FORECAST_HORIZON:
        raise ValueError(f"Estimator must have {FORECAST_HORIZON} direct outputs")
    if bundle.get("target_mode") not in ("absolute", "delta"):
        raise ValueError("Invalid WTI target mode")
    profile = bundle.get("daily_profile")
    if not isinstance(profile, pd.DataFrame) or list(profile.columns) != list(MINUTE_COLUMNS):
        raise ValueError("Model is missing the training-only daily reference")
    if list(profile.index) != list(range(24 * 60 // REFERENCE_INTERVAL_MINUTES)) or not np.isfinite(profile.to_numpy()).all():
        raise ValueError("Daily reference must cover all time slots with finite values")
    return bundle


def predict_values(bundle, features):
    """Shared train/inference conversion from direct outputs to absolute raw WTI."""
    with threadpool_limits(limits=1):
        values = np.asarray(bundle["model"].predict(features), dtype=float)
    if values.shape != (len(features), FORECAST_HORIZON) or not np.isfinite(values).all():
        raise ValueError(f"Model must return {FORECAST_HORIZON} finite WTI values per row")
    if bundle["target_mode"] == "delta":
        values = values + features[TARGET_COLUMN].to_numpy()[:, None]
    return np.maximum(values, 0)


def provisional_values(history, profile, times):
    """Causal, damped level/trend correction; synthetic history never enters sklearn.

    Fixed conservative defaults, not fitted to data2. Thirty minutes estimate the
    residual trend; the latest three set the level. Corrections decay with horizon.
    """
    mask = history.get("is_reference", pd.Series(False, index=history.index))
    valid = ~mask & history[TARGET_COLUMN].notna() & (history.samples_per_minute > 0)
    # Use only the contiguous real suffix; do not bridge a camera outage.
    last_invalid = np.flatnonzero(~valid.to_numpy())
    real = history.iloc[last_invalid[-1] + 1:] if len(last_invalid) else history
    real = real.tail(30)
    values = reference_minutes(profile, times)[TARGET_COLUMN].to_numpy()
    if real.empty:
        return values
    residual = real[TARGET_COLUMN].to_numpy() - reference_minutes(profile, real.index)[TARGET_COLUMN].to_numpy()
    level = float(np.mean(residual[-SMOOTHING_WINDOW:]))
    slope = 0.0
    if len(real) >= 10:
        # Robust endpoints reduce the influence of individual detection spikes.
        slope = (np.median(residual[-5:]) - np.median(residual[:5])) / (len(real) - 5)
        slope = float(np.clip(slope, -1.0, 1.0))
    steps = np.arange(1, len(times) + 1)
    decay = np.exp(-steps / 20.0)
    correction = level + slope * 10 * (1 - np.exp(-steps / 10.0))
    return np.maximum(values + correction * decay, 0)


class TrafficForecaster:
    """Load once; failures become displayable results without crashing the UI."""

    def __init__(self, model_path=MODEL_PATH):
        self.bundle = None
        self.error = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Trying to unpickle estimator.*", category=UserWarning)
                self.bundle = validate_bundle(joblib.load(model_path))
        except Exception as error:
            if hasattr(error, "original_sklearn_version"):
                self.error = (f"Model dùng sklearn {error.original_sklearn_version}, môi trường đang chạy "
                              f"{error.current_sklearn_version}. Mở app bằng đúng môi trường đã train; không cần train lại.")
            else:
                self.error = f"Chưa tải được model WTI: {error}. Kiểm tra file model và môi trường đã train."

    def predict(self, history: pd.DataFrame, daily_profile=None) -> ForecastResult:
        if len(history) < HISTORY_POINTS:
            return ForecastResult(status_message=(
                f"Chưa đủ dữ liệu để dự báo. Cần {HISTORY_POINTS} phút hoàn tất (gồm lag 60)."
            ))
        try:
            history = regular_minutes(history).tail(HISTORY_POINTS)
            if len(history) != HISTORY_POINTS or not np.isfinite(history[TARGET_COLUMN].iloc[-1]) or history.samples_per_minute.iloc[-1] <= 0:
                return ForecastResult(status_message=(
                    "Phút hiện tại chưa có mẫu Camera 1; tạm dừng dự báo, không điền dữ liệu giả."
                ))
            times = pd.date_range(
                history.index[-1] + pd.Timedelta(minutes=1),
                periods=FORECAST_HORIZON, freq=RESAMPLE_INTERVAL,
            )
            mask = history.get("is_reference", pd.Series(False, index=history.index)).fillna(False).astype(bool)
            references = int(mask.sum())
            valid = ~mask & history[TARGET_COLUMN].notna() & (history.samples_per_minute > 0)
            real_count = int(valid.sum())
            sparse = int((valid & (history.samples_per_minute < MIN_SAMPLES_PER_MINUTE)).sum())
            provisional = real_count < HISTORY_POINTS or sparse > 0 or self.bundle is None
            if provisional:
                profile = daily_profile if daily_profile is not None else (
                    self.bundle["daily_profile"] if self.bundle is not None else None
                )
                if profile is None:
                    return ForecastResult(alert_level="MODEL_ERROR", status_message=self.error or "Thiếu đường WTI tham chiếu.")
                values = provisional_values(history, profile, times)
                real = history.loc[~mask].tail(SMOOTHING_WINDOW)
                source_message = (
                    f"TẠM THỜI · Độ tin cậy thấp · Tham chiếu hiệu chỉnh mức/xu hướng realtime; "
                    f"{real_count}/{HISTORY_POINTS} phút thật, {references} phút tham chiếu."
                    if not real.empty else "TẠM THỜI · Chưa có phút realtime hoàn tất; dùng WTI thường nhật tham chiếu."
                )
                if self.error:
                    source_message += " Chưa có model WTI dùng được. " + self.error
            else:
                features = create_features(history).iloc[[-1]]
                if not np.isfinite(features.to_numpy()).all():
                    return ForecastResult(status_message="Chưa đủ dữ liệu để tạo feature WTI.")
                values = predict_values(self.bundle, features)[0]
                source_message = "MODEL WTI · 61 phút Camera 1 thật; dự báo trực tiếp 60 phút."
            raw_values = np.maximum(values, 0)
            past = history.loc[~mask, TARGET_COLUMN].tail(2)
            if past.empty:
                past = history[TARGET_COLUMN].tail(2)
            values = smooth_forecast(raw_values, past)
            eta = stable_crossing(values)
            # WARNING: stable exceedance starts within 5 minutes. WATCH: later.
            level = "NORMAL" if eta is None else ("WARNING" if eta <= 5 else "WATCH")
            if provisional and eta is None:
                level = "PROVISIONAL"
            # Distinguish a currently observed stable exceedance from a future onset.
            recent = history.iloc[-5:]
            if valid.iloc[-5:].all() and stable_crossing(recent[TARGET_COLUMN]) == 1:
                eta, level = 0, "WARNING"
            message = ("Chưa dự báo thấy vượt ngưỡng trong 60 phút; không bảo đảm giao thông thông thoáng."
                       if eta is None else f"Cửa sổ vượt ngưỡng thử nghiệm bắt đầu sau {eta} phút (quy tắc 3/5).")
            if eta == 0:
                message = "Đang vượt ngưỡng WTI thử nghiệm ổn định (3/5 phút thật)."
            return ForecastResult(
                forecast_times=times, forecast_values=values, threshold=CONGESTION_THRESHOLD,
                eta_minutes=eta, crossing_time=None if eta is None else (history.index[-1] if eta == 0 else times[eta - 1]),
                alert_level=level, is_ready=True, status_message=source_message + " " + message,
                reference_minutes=references, is_provisional=provisional, raw_forecast_values=raw_values,
                real_minutes=real_count, model_error=self.error,
                quality_message=f"{real_count}/61 phút thật · {sparse} phút dưới {MIN_SAMPLES_PER_MINUTE} mẫu · tuổi ảnh nguồn chưa xác minh",
            )
        except Exception as error:
            return ForecastResult(alert_level="MODEL_ERROR", status_message=f"Lỗi dự báo: {error}")
