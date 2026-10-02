"""WTI backtest: 2-hour past display, 60 future minutes, optional startup replay."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from core.traffic_features import (
    DISPLAY_WINDOW, MINUTE_COLUMNS, MODEL_PATH, TARGET_COLUMN,
    build_daily_profile, read_minutes, reference_minutes, smooth_observed, supervised_data,
)
from core.traffic_forecast import TrafficForecaster, stable_crossing


def backtest(data_path, model_path, at, output, warmup_minutes=None):
    minutes = read_minutes(data_path)
    features, targets = supervised_data(minutes)
    if features.empty:
        raise ValueError("No complete history/forecast windows")
    if at is None:
        # Default to an approaching observed threshold event, exposing missed peaks.
        candidates = [time for time, row in targets.iterrows() if (stable_crossing(row) or 0) >= 5]
        origin = candidates[0] if candidates else features.index[len(features) // 2]
    else:
        origin = pd.Timestamp(at)
        if origin not in features.index:
            raise ValueError("--at must be a complete minute with 61 past and 60 future observations")
    history = minutes.loc[:origin].tail(DISPLAY_WINDOW + 1).copy()
    profile = None
    if warmup_minutes is not None:
        profile = build_daily_profile(read_minutes(ROOT / "data/data1.csv"))
        times = pd.date_range(origin - pd.Timedelta(minutes=DISPLAY_WINDOW), origin, freq="1min")
        history = reference_minutes(profile, times)
        history["is_reference"] = True
        if warmup_minutes:
            live = times[-warmup_minutes:]
            history.loc[live, list(MINUTE_COLUMNS)] = minutes.loc[live, list(MINUTE_COLUMNS)]
            history.loc[live, "is_reference"] = False
    result = TrafficForecaster(model_path).predict(history, profile)
    if not result.is_ready:
        raise ValueError(result.status_message)
    future = targets.loc[origin].to_numpy()
    actual_eta = stable_crossing(future)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    actual = smooth_observed(history)
    mask = history.get("is_reference", pd.Series(False, index=history.index))
    ax.plot(actual.index, actual.where(~mask), color="#2563eb", label="Observed WTI / trailing 3 min")
    ax.plot(history.index, history[TARGET_COLUMN].where(mask), color="#94a3b8", linestyle=":", label="Reference past (not observations)")
    label = "Provisional calibrated reference / 3 min" if result.is_provisional else "Direct WTI model / 3 min"
    ax.plot([origin] + list(result.forecast_times), [actual.iloc[-1]] + list(result.forecast_values), color="#f59e0b", linewidth=2, label=label)
    ax.plot(result.forecast_times, future, color="#16a34a", label="Held-out RAW WTI next 60 min")
    ax.axhline(result.threshold, color="#dc2626", linestyle="--", label="Experimental threshold")
    ax.axvline(origin, color="#64748b", linestyle=":", label="Forecast origin")
    if result.crossing_time is not None:
        ax.axvline(result.crossing_time, color="#dc2626", linestyle="-.", label="Predicted stable-window start")
    if actual_eta is not None:
        ax.axvline(result.forecast_times[actual_eta - 1], color="#16a34a", linestyle="-.", label="Actual stable-window start")
    predicted_text = "none within 60 min" if result.eta_minutes is None else f"{result.eta_minutes} min"
    actual_text = "none within 60 min" if actual_eta is None else f"{actual_eta} min"
    ax.set_title(f"WTI backtest at {origin} / {'PROVISIONAL' if result.is_provisional else 'MODEL'}\nPredicted ETA: {predicted_text} | Actual ETA (raw WTI): {actual_text} (3 of 5 rule)")
    ax.set_xlabel("Vietnam local time / completed minute")
    ax.set_ylabel("WTI / weighted vehicles per image")
    ax.set_xlim(origin - pd.Timedelta(minutes=DISPLAY_WINDOW), origin + pd.Timedelta(minutes=DISPLAY_WINDOW))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    info = {
        "origin": str(origin), "predicted_eta_minutes": result.eta_minutes,
        "actual_eta_minutes": actual_eta, "threshold": result.threshold,
        "forecast_times": [str(time) for time in result.forecast_times],
        "forecast_values": result.forecast_values.tolist(), "future_actual": future.tolist(),
        "raw_forecast_values": result.raw_forecast_values.tolist(), "is_provisional": result.is_provisional,
        "warmup_minutes": warmup_minutes, "target_column": TARGET_COLUMN,
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Backtest: {output}; predicted ETA={predicted_text}; actual ETA={actual_text}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data2.csv")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--at", help="Completed minute, e.g. '2026-09-29 09:00'")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/wti_backtest.png")
    parser.add_argument("--warmup-minutes", type=int, choices=range(0, 61), help="Replay startup with 0..60 real minutes; no trained model needed")
    args = parser.parse_args()
    backtest(args.data, args.model, args.at, args.output, args.warmup_minutes)


if __name__ == "__main__":
    main()
