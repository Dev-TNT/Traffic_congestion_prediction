"""Score previously issued forecasts against subsequently observed Camera 1 WTI.

Read-only: does not train, modify the database, or fill missing ground truth.
"""

import argparse
import json
from pathlib import Path
import runpy
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.traffic_features import TARGET_COLUMN, MIN_SAMPLES_PER_MINUTE, resample_minutes
from core.traffic_store import STORE_PATH


def evaluate_store(path):
    evaluate = runpy.run_path(str(ROOT / "training_model.Py"))["evaluate"]
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        samples = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM samples ORDER BY timestamp")]
        forecasts = connection.execute("SELECT origin, issued_at, payload FROM forecasts ORDER BY origin, issued_at").fetchall()
    finally:
        connection.close()
    if not samples:
        return {"status": "No observed samples yet", "groups": {}}
    minutes = resample_minutes(pd.DataFrame(samples))
    groups, seen = {}, set()
    skipped = 0
    for origin, issued_at, payload in forecasts:
        if origin in seen:
            continue  # first issued forecast only; never cherry-pick late revisions
        seen.add(origin)
        prediction = json.loads(payload)
        times = pd.DatetimeIndex(prediction["times"])
        if len(times) != 60 or pd.Timestamp(issued_at) >= times[0]:
            skipped += 1
            continue
        truth = minutes.reindex(times)
        # A currently unfinished or poorly sampled minute is not ground truth.
        if (times[-1] > pd.Timestamp.now().floor("min") or
                truth[TARGET_COLUMN].isna().any() or (truth.samples_per_minute < MIN_SAMPLES_PER_MINUTE).any()):
            skipped += 1
            continue
        key = f"{prediction['method']} / {prediction['model_id']} / threshold={prediction['threshold']}"
        actual, predicted, threshold = groups.setdefault(key, ([], [], prediction["threshold"]))
        actual.append(truth[TARGET_COLUMN].to_numpy())
        predicted.append(prediction["values"])
    return {
        "unit": "overlapping forecast-origin windows, experimental WTI threshold; not congestion episodes",
        "skipped_incomplete_or_late": skipped,
        "groups": {name: evaluate(np.array(actual), np.array(predicted), threshold)
                   for name, (actual, predicted, threshold) in groups.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=STORE_PATH)
    args = parser.parse_args()
    if not args.database.exists():
        parser.exit(message="No history yet. Run the app to collect Camera 1 observations.\n")
    print(json.dumps(evaluate_store(args.database), indent=2, ensure_ascii=False))
