"""Camera 1 observations and issued forecasts, owned by the forecast worker."""

import json
from pathlib import Path
import sqlite3

import pandas as pd

from core.traffic_features import MODEL_PATH


STORE_PATH = MODEL_PATH.parents[1] / "runs/traffic_history.sqlite3"


class TrafficStore:
    def __init__(self, path=STORE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=2)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                timestamp TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS forecasts (
                origin TEXT NOT NULL, issued_at TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (origin, issued_at)
            );
        """)

    def save_sample(self, sample):
        row = dict(sample)
        row["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO samples VALUES (?, ?)",
                                    (row["timestamp"], json.dumps(row)))

    def samples_since(self, cutoff):
        rows = self.connection.execute(
            "SELECT payload FROM samples WHERE timestamp >= ? ORDER BY timestamp",
            (pd.Timestamp(cutoff).isoformat(),),
        )
        return [json.loads(row[0]) for row in rows]

    def save_forecast(self, origin, result, model_id):
        if not result.is_ready:
            return
        payload = {
            "times": [time.isoformat() for time in result.forecast_times],
            "values": result.forecast_values.tolist(),
            "raw_values": result.raw_forecast_values.tolist(),
            "threshold": result.threshold, "eta_minutes": result.eta_minutes,
            "provisional": result.is_provisional, "model_id": model_id,
            "method": "damped-residual-v1" if result.is_provisional else "model-3min",
            "real_minutes": result.real_minutes, "quality": result.quality_message,
        }
        with self.connection:
            self.connection.execute("INSERT INTO forecasts VALUES (?, ?, ?)",
                                    (pd.Timestamp(origin).isoformat(), pd.Timestamp.now().isoformat(),
                                     json.dumps(payload)))

    def close(self):
        self.connection.close()
