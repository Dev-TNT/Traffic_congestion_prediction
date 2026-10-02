"""Shared, causal minute aggregation and features for training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd


# Experimental raw WTI threshold; not a calibrated congestion label.
CONGESTION_THRESHOLD = 100.0
FORECAST_HORIZON = 60
HISTORY_WINDOW = 60
HISTORY_POINTS = HISTORY_WINDOW + 1  # current minute plus lag 60
DISPLAY_WINDOW = 120
SMOOTHING_WINDOW = 3
MIN_SAMPLES_PER_MINUTE = 3  # minimum observed coverage for model input/scoring
RESAMPLE_INTERVAL = "1min"
MODEL_VERSION = "traffic-wti-v1"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/traffic_wti_forecast.joblib"
TARGET_COLUMN = "WTI_mean"
FORECAST_CAMERA = "Camera 1"
REFERENCE_INTERVAL_MINUTES = 15
LAGS = (1, 2, 3, 5, 10, 15, 30, 60)
ROLLING_WINDOWS = (5, 15, 30, 60)
MINUTE_COLUMNS = (
    "car_mean", "motorbike_mean", "bus_mean", "truck_mean", "total_mean",
    "WTI_mean", "total_std", "total_max", "WTI_std", "WTI_max", "samples_per_minute",
)
FEATURE_NAMES = (
    list(MINUTE_COLUMNS)
    + [name for signal in ("total", "WTI") for name in (
        [f"{signal}_lag_{lag}" for lag in LAGS]
        + [f"{signal}_mean_{window}" for window in ROLLING_WINDOWS]
        + [f"{signal}_std_{window}" for window in (5, 15, 30)]
        + [f"{signal}_{stat}_{window}" for stat in ("min", "max") for window in (5, 15, 30)]
        + [f"{signal}_slope_{window}" for window in (5, 15, 30)]
    )]
    + ["hour_sin", "hour_cos"]
)


def validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Validate counts; keep CSV timestamps as naive Vietnam local time."""
    required = ["timestamp", "car", "motorbike", "bus", "truck", "total", "WTI"]
    missing = set(required) - set(samples.columns)
    if missing:
        raise ValueError(f"Missing sample columns: {sorted(missing)}")
    result = samples[required].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    if result["timestamp"].isna().any() or result["timestamp"].dt.tz is not None:
        raise ValueError("Timestamps must be valid, timezone-naive local times")
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate sample timestamps")
    numeric = result[required[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError("Counts and WTI must be finite and nonnegative")
    counts = numeric[["car", "motorbike", "bus", "truck", "total"]]
    if not np.equal(counts, np.floor(counts)).all().all():
        raise ValueError("Raw vehicle counts must be integers")
    if not np.allclose(numeric["total"], counts.iloc[:, :4].sum(axis=1)):
        raise ValueError("total must equal car + motorbike + bus + truck")
    expected_wti = 4 * counts.car + counts.motorbike + 7 * counts.bus + 6 * counts.truck
    if not np.allclose(numeric["WTI"], expected_wti):
        raise ValueError("WTI does not match the detector's 4/1/7/6 weights")
    result[required[1:]] = numeric
    return result.sort_values("timestamp")


def read_samples(path) -> pd.DataFrame:
    return validate_samples(pd.read_csv(path))


def read_minutes(path) -> pd.DataFrame:
    samples = read_samples(path)
    minutes = resample_minutes(samples)
    # CSV boundaries may contain partial minutes: exclude them from evaluation.
    start = samples.timestamp.min().ceil(RESAMPLE_INTERVAL)
    end = samples.timestamp.max().floor(RESAMPLE_INTERVAL)
    return minutes.loc[(minutes.index > start) & (minutes.index <= end)]


def resample_minutes(samples: pd.DataFrame) -> pd.DataFrame:
    """Label [10:00, 10:01) at 10:01, when that minute becomes available.

    Missing minutes remain NaN with zero samples. A one-sample std is zero.
    Callers must exclude the currently unfinished minute.
    """
    rows = validate_samples(samples).set_index("timestamp")
    minutes = rows.resample(RESAMPLE_INTERVAL, closed="left", label="right").agg(
        car_mean=("car", "mean"), motorbike_mean=("motorbike", "mean"),
        bus_mean=("bus", "mean"), truck_mean=("truck", "mean"),
        total_mean=("total", "mean"), WTI_mean=("WTI", "mean"),
        total_std=("total", lambda values: values.std(ddof=0)),
        total_max=("total", "max"), samples_per_minute=("total", "count"),
        WTI_std=("WTI", lambda values: values.std(ddof=0)), WTI_max=("WTI", "max"),
    )
    return minutes.loc[:, list(MINUTE_COLUMNS)]


def regular_minutes(history: pd.DataFrame) -> pd.DataFrame:
    if set(MINUTE_COLUMNS) - set(history.columns):
        raise ValueError("Minute history schema does not match MINUTE_COLUMNS")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Minute history requires a DatetimeIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and increasing")
    if not history.index.equals(history.index.floor(RESAMPLE_INTERVAL)):
        raise ValueError("Minute timestamps must align to minute boundaries")
    return history.asfreq(RESAMPLE_INTERVAL)


def create_features(history: pd.DataFrame) -> pd.DataFrame:
    """Only current/past observations; invalidate windows crossing missing data."""
    minutes = regular_minutes(history)
    total = minutes["total_mean"].where(minutes["samples_per_minute"] > 0)
    features = minutes[list(MINUTE_COLUMNS)].copy()
    for signal in ("total", "WTI"):
        values = minutes[f"{signal}_mean"].where(minutes.samples_per_minute > 0)
        for lag in LAGS:
            features[f"{signal}_lag_{lag}"] = values.shift(lag)
        for window in ROLLING_WINDOWS:
            rolling = values.rolling(window, min_periods=window)
            features[f"{signal}_mean_{window}"] = rolling.mean()
            if window != HISTORY_WINDOW:
                features[f"{signal}_std_{window}"] = rolling.std(ddof=0)
                features[f"{signal}_min_{window}"] = rolling.min()
                features[f"{signal}_max_{window}"] = rolling.max()
                features[f"{signal}_slope_{window}"] = (values - values.shift(window)) / window
    hour = minutes.index.hour + minutes.index.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    complete = total.rolling(HISTORY_POINTS, min_periods=HISTORY_POINTS).count()
    features.loc[complete != HISTORY_POINTS, :] = np.nan
    return features[FEATURE_NAMES]


def create_targets(history: pd.DataFrame) -> pd.DataFrame:
    minutes = regular_minutes(history)
    values = minutes[TARGET_COLUMN].where(minutes.samples_per_minute > 0)
    return pd.DataFrame({
        f"WTI_t+{step}": values.shift(-step)
        for step in range(1, FORECAST_HORIZON + 1)
    }, index=minutes.index)


def supervised_data(history: pd.DataFrame):
    features, targets = create_features(history), create_targets(history)
    valid = features.notna().all(axis=1) & targets.notna().all(axis=1)
    return features.loc[valid], targets.loc[valid]


def build_daily_profile(train_minutes: pd.DataFrame) -> pd.DataFrame:
    """Fit a provisional 15-minute daily reference on TRAIN observations only."""
    observed = train_minutes.loc[train_minutes.samples_per_minute > 0, list(MINUTE_COLUMNS)]
    slots = (observed.index.hour * 60 + observed.index.minute) // REFERENCE_INTERVAL_MINUTES
    profile = observed.groupby(slots).mean().reindex(range(24 * 60 // REFERENCE_INTERVAL_MINUTES))
    if not np.isfinite(profile.to_numpy()).all():
        raise ValueError("Training capture does not cover the complete daily reference")
    return profile


def reference_minutes(profile: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Map a training-only reference to clock times without using live future data."""
    slots = (times.hour * 60 + times.minute) // REFERENCE_INTERVAL_MINUTES
    return pd.DataFrame(profile.loc[slots, list(MINUTE_COLUMNS)].to_numpy(), index=times, columns=MINUTE_COLUMNS)


def smooth_forecast(values, past_values):
    """Trailing three-point mean, seeded only with available past observations."""
    prefix = np.asarray(past_values, dtype=float)[-(SMOOTHING_WINDOW - 1):]
    joined = np.concatenate([prefix, np.asarray(values, dtype=float)])
    return pd.Series(joined).rolling(SMOOTHING_WINDOW, min_periods=1).mean().to_numpy()[len(prefix):]


def smooth_observed(history):
    """Smooth only real WTI segments; do not blend references or camera outages."""
    references = history.get("is_reference", pd.Series(False, index=history.index))
    values = history[TARGET_COLUMN].where(~references)
    groups = values.isna().cumsum()
    smoothed = values.groupby(groups).transform(
        lambda segment: segment.rolling(SMOOTHING_WINDOW, min_periods=1).mean()
    ).where(values.notna())
    return smoothed.where(~references, history[TARGET_COLUMN])
