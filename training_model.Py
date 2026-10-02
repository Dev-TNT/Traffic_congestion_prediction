"""Train raw WTI t+1..t+60. Import and --describe NEVER train.

Read build_model() for the architecture. Two target modes are selected using
chronological data1 validation with a 60-minute gap. data2 is final test only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from core.traffic_features import (
    CONGESTION_THRESHOLD, FEATURE_NAMES, FORECAST_HORIZON, HISTORY_POINTS,
    HISTORY_WINDOW, MINUTE_COLUMNS, MODEL_PATH, MODEL_VERSION, RESAMPLE_INTERVAL,
    REFERENCE_INTERVAL_MINUTES, SMOOTHING_WINDOW, TARGET_COLUMN,
    build_daily_profile, read_minutes, reference_minutes, smooth_forecast, supervised_data,
)
from core.traffic_forecast import predict_values, provisional_values, stable_crossing, validate_bundle

ROOT = Path(__file__).resolve().parent


def build_model(max_iter=200):
    # 60 separate direct regressors; no recursion or random internal validation.
    return MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=25,
        learning_rate=0.05, l2_regularization=5.0,
        early_stopping=False, random_state=42,
    ), n_jobs=1)


def chronological_split(features):
    validation_start = features.index[int(len(features) * 0.8)]
    fit = features.index + pd.Timedelta(minutes=FORECAST_HORIZON) < validation_start
    validation = features.index >= validation_start
    if fit.sum() < 100 or validation.sum() < 30:
        raise ValueError("Too few time-ordered fit/validation windows")
    return fit, validation


def evaluate(actual, predicted, threshold=CONGESTION_THRESHOLD):
    actual, predicted = np.asarray(actual), np.maximum(np.asarray(predicted), 0)
    actual_eta = np.array([stable_crossing(row, threshold) or np.nan for row in actual])
    predicted_eta = np.array([stable_crossing(row, threshold) or np.nan for row in predicted])
    observed, warned = np.isfinite(actual_eta), np.isfinite(predicted_eta)
    tp = int(np.count_nonzero(observed & warned))
    fp = int(np.count_nonzero(~observed & warned))
    fn = int(np.count_nonzero(observed & ~warned))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mae = np.abs(actual - predicted).mean(axis=0)
    return {
        "mae_by_horizon": {str(h): float(value) for h, value in enumerate(mae, 1)},
        "mean_mae": float(mae.mean()),
        "threshold_events": {
            "unit": "forecast-origin windows (overlap; not independent episodes)",
            "evaluated_origins": len(actual), "actual_positive_origins": int(observed.sum()),
            "true_positives": tp, "false_alarms": fp, "missed_events": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "eta_matched_origins": tp,
            "eta_mae_minutes": float(np.abs(actual_eta[observed & warned] - predicted_eta[observed & warned]).mean()) if tp else None,
        },
    }


def smooth_predictions(predictions, minutes, origins):
    return np.array([smooth_forecast(row, minutes.loc[:origin, TARGET_COLUMN].tail(2))
                     for origin, row in zip(origins, predictions)])


def baselines(features, profile=None):
    current = features[TARGET_COLUMN].to_numpy()[:, None]
    steps = np.arange(1, FORECAST_HORIZON + 1)
    result = {
        "persistence": np.repeat(current, FORECAST_HORIZON, axis=1),
        "linear_trend_15min": np.maximum(current + features.WTI_slope_15.to_numpy()[:, None] * steps, 0),
    }
    if profile is not None:
        times = pd.DatetimeIndex((features.index.to_numpy()[:, None] + steps * np.timedelta64(1, "m")).ravel())
        result["same_time_train_15min"] = reference_minutes(profile, times)[TARGET_COLUMN].to_numpy().reshape(len(features), -1)
    return result


def training_targets(targets, features, mode):
    return targets.subtract(features[TARGET_COLUMN], axis=0) if mode == "delta" else targets


def choose_output_path(output):
    output = Path(output)
    if output.exists() or output.with_suffix(".metrics.json").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return output.with_name(f"{output.stem}.candidate-{stamp}{output.suffix}")
    return output


def train(train_path, test_path, output, max_iter=200):
    train_minutes, test_minutes = read_minutes(train_path), read_minutes(test_path)
    if train_minutes.empty or test_minutes.empty:
        raise ValueError("Captures contain no complete minutes")
    gap = (test_minutes.index[0] - train_minutes.index[-1]).total_seconds() / 60
    if gap <= FORECAST_HORIZON:
        raise ValueError("Test capture must start more than 60 minutes after train ends")
    x_train, y_train = supervised_data(train_minutes)
    x_test, y_test = supervised_data(test_minutes)
    if len(x_train) < 200 or x_test.empty:
        raise ValueError("Too few complete WTI history/target windows")
    fit, validation = chronological_split(x_train)
    validation_metrics = {}
    for mode in ("absolute", "delta"):
        print(f"Validation candidate: {mode}", flush=True)
        model = build_model(max_iter)
        with threadpool_limits(limits=1):
            model.fit(x_train.loc[fit], training_targets(y_train.loc[fit], x_train.loc[fit], mode))
        prediction = predict_values({"model": model, "target_mode": mode}, x_train.loc[validation])
        validation_metrics[mode] = evaluate(y_train.loc[validation], smooth_predictions(prediction, train_minutes, x_train.index[validation]))
    selected_mode = min(validation_metrics, key=lambda mode: validation_metrics[mode]["mean_mae"])
    for name, prediction in baselines(x_train.loc[validation]).items():
        validation_metrics[name] = evaluate(y_train.loc[validation], smooth_predictions(prediction, train_minutes, x_train.index[validation]))
    print(f"Selected on data1 only: {selected_mode}. Refitting all data1...", flush=True)
    model = build_model(max_iter)
    with threadpool_limits(limits=1):
        model.fit(x_train, training_targets(y_train, x_train, selected_mode))
    profile = build_daily_profile(train_minutes)
    raw = predict_values({"model": model, "target_mode": selected_mode}, x_test)
    deployed = smooth_predictions(raw, test_minutes, x_test.index)
    metrics = {"model_raw": evaluate(y_test, raw), "model_3min": evaluate(y_test, deployed)}
    for name, prediction in baselines(x_test, profile).items():
        metrics[name + "_3min"] = evaluate(y_test, smooth_predictions(prediction, test_minutes, x_test.index))
    # Evaluate realistic startup transitions. Synthetic minutes never enter sklearn.
    for real_minutes in (0, 1, 3, 5, 15, 30, 60):
        predictions = []
        origins = x_test.index[::30]
        for origin in origins:
            times = pd.date_range(origin - pd.Timedelta(minutes=HISTORY_WINDOW), origin, freq=RESAMPLE_INTERVAL)
            history = reference_minutes(profile, times)
            history["is_reference"] = True
            if real_minutes:
                live_times = times[-real_minutes:]
                history.loc[live_times, list(MINUTE_COLUMNS)] = test_minutes.loc[live_times, list(MINUTE_COLUMNS)]
                history.loc[live_times, "is_reference"] = False
            future_times = pd.date_range(origin + pd.Timedelta(minutes=1), periods=FORECAST_HORIZON, freq=RESAMPLE_INTERVAL)
            values = provisional_values(history, profile, future_times)
            real = history.loc[~history.is_reference, TARGET_COLUMN].tail(2)
            predictions.append(smooth_forecast(values, real if len(real) else history[TARGET_COLUMN].tail(2)))
        metrics[f"provisional_{real_minutes}_real_minutes_3min"] = evaluate(y_test.loc[origins], predictions)
    metrics["model_3min_nonoverlapping_60min"] = evaluate(y_test.iloc[::60], deployed[::60])
    report = {
        "model_version": MODEL_VERSION, "target_column": TARGET_COLUMN,
        "forecast_horizon": FORECAST_HORIZON, "congestion_threshold": CONGESTION_THRESHOLD,
        "smoothing_window": SMOOTHING_WINDOW, "selected_target_mode": selected_mode,
        "validation": {"metrics": validation_metrics, "fit_origins": int(fit.sum()),
                       "validation_origins": int(validation.sum()),
                       "last_fit_origin": str(x_train.index[fit][-1]),
                       "first_validation_origin": str(x_train.index[validation][0]),
                       "target_gap_minutes": FORECAST_HORIZON},
        "split_gap_minutes": gap, "metrics": metrics,
        "data": {name: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        "start": str(minutes.index[0]), "end": str(minutes.index[-1]),
                        "minute_rows": len(minutes), "supervised_origins": count,
                        "missing_minutes": int((minutes.samples_per_minute == 0).sum())}
                 for name, path, minutes, count in (("train", train_path, train_minutes, len(x_train)),
                                                     ("test", test_path, test_minutes, len(x_test)))},
        "limitations": "Two captures only; WTI threshold uncalibrated. Smoothed prediction MAE uses RAW minute WTI truth. No accuracy guarantee or congestion ground truth.",
    }
    bundle = {
        "model": model, "target_mode": selected_mode, "target_column": TARGET_COLUMN,
        "feature_names": FEATURE_NAMES, "forecast_horizon": FORECAST_HORIZON,
        "resample_interval": RESAMPLE_INTERVAL, "history_window": HISTORY_WINDOW,
        "history_points": HISTORY_POINTS, "congestion_threshold": CONGESTION_THRESHOLD,
        "model_version": MODEL_VERSION, "smoothing_window": SMOOTHING_WINDOW,
        "daily_profile": profile, "reference_interval_minutes": REFERENCE_INTERVAL_MINUTES,
        "metrics": metrics, "training_report": report,
        "sklearn_version": sklearn.__version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    validate_bundle(bundle)
    requested_output = Path(output)
    output = choose_output_path(requested_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output != requested_output:
        print("Existing model preserved; saving a candidate for review.")
    with output.open("xb") as handle:
        joblib.dump(bundle, handle, compress=3)
    reloaded = validate_bundle(joblib.load(output))
    np.testing.assert_allclose(predict_values(reloaded, x_test.iloc[:1]), raw[:1])
    with output.with_suffix(".metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Train/test origins: {len(x_train)}/{len(x_test)}; gap: {gap:.0f} minutes")
    print("Metric (WTI)                          +1     +5    +10    +15    +30    +60   Mean")
    for name, values in metrics.items():
        maes = values["mae_by_horizon"]
        print(f"{name:36s}" + "".join(f"{maes[str(h)]:7.2f}" for h in (1, 5, 10, 15, 30, 60)) + f"{values['mean_mae']:7.2f}")
        print("  Threshold/ETA:", values["threshold_events"])
    if metrics["model_3min"]["mean_mae"] >= metrics["persistence_3min"]["mean_mae"]:
        print("Model does not beat persistence on this held-out capture; inspect backtest before use.")
    print(f"Saved and reloaded: {output}")
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/data1.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/data2.csv")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--describe", action="store_true", help="Print architecture/configuration without training")
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    print(build_model(args.max_iter))
    print(f"Target={TARGET_COLUMN}; direct horizons=1..{FORECAST_HORIZON}; history={HISTORY_POINTS}; features={len(FEATURE_NAMES)}; smoothing={SMOOTHING_WINDOW}; threshold={CONGESTION_THRESHOLD}")
    print("Features:", ", ".join(FEATURE_NAMES))
    print("Target modes: absolute WTI / delta from current WTI; data1 validation gap=60. data2 is final test only.")
    if not args.describe:
        train(args.train, args.test, args.output, args.max_iter)


if __name__ == "__main__":
    main()
