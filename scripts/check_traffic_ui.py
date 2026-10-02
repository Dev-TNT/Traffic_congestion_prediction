"""Bounded full-window verification with historical replay or the real camera.

Replay uses the real forecast worker and queued count API, with an explicit
historical clock. Live mode uses the unchanged YOLO image_processing function.
Screenshots are captured from the actual Qt window, not an HTML mockup.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from PyQt5.QtCore import Q_ARG, QMetaObject, QTimer, Qt
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from core.traffic_features import DISPLAY_WINDOW, FORECAST_CAMERA, FORECAST_HORIZON, MODEL_PATH, read_samples
from core.vision_worker import CameraWorker
from ui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use real camera/YOLO instead of CSV replay")
    parser.add_argument("--startup", action="store_true", help="Show initial reference history instead of CSV replay")
    parser.add_argument("--reference", action="store_true", help="Enable the daily comparison button")
    parser.add_argument("--dark", action="store_true", help="Verify contrast in the dark theme")
    parser.add_argument("--warmup-minutes", type=int, choices=range(0, 61), help="Replay only 0..60 real minutes to verify provisional calibration")
    parser.add_argument("--viewer", default="Camera 2", help="Viewer camera in live verification (forecast remains Camera 1)")
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--at", default="2026-09-29 09:15")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    setTheme(Theme.DARK if args.dark else Theme.LIGHT)
    window = MainWindow()
    dashboard = window.dashboardTab
    app.aboutToQuit.connect(dashboard.stop)
    window.switchTo(dashboard)
    if args.live:
        window.cameraTab.cameraCombo.setCurrentText(args.viewer)
    if args.reference:
        dashboard.reference_button.click()
    window.show()
    counts, frames, heartbeats, updates = [], [], [], []
    report = {"mode": "live" if args.live else ("startup reference" if args.startup else "historical replay")}
    report["initial_viewer_camera"] = window.cameraTab.cameraCombo.currentText()
    report["initial_viewer_detection_enabled"] = window.cameraTab.yoloSwitch.isChecked()
    viewer_changes = []
    window.cameraTab.cameraCombo.currentTextChanged.connect(lambda value: viewer_changes.append({"camera": value}))
    window.cameraTab.yoloSwitch.checkedChanged.connect(lambda value: viewer_changes.append({"detect": value}))
    camera_worker = None
    if args.live:
        camera_worker = CameraWorker(app)
        camera_worker.frame_ready.connect(window.cameraTab.update_frame)
        camera_worker.frame_ready.connect(lambda image: frames.append(not image.isNull()))
        camera_worker.counts_ready.connect(dashboard.receive_sample)
        camera_worker.counts_ready.connect(counts.append)
        app.aboutToQuit.connect(camera_worker.stop)
        camera_worker.start()
    elif not args.startup:
        report["forecast_origin"] = args.at
        window.setWindowTitle("Traffic Jam Prediction · HISTORICAL REPLAY (data2)")
        rows = read_samples(ROOT / "data/data2.csv")
        origin = pd.Timestamp(args.at)
        lookback = DISPLAY_WINDOW + 1 if args.warmup_minutes is None else args.warmup_minutes
        rows = rows.loc[(rows.timestamp >= origin - pd.Timedelta(minutes=lookback)) & (rows.timestamp < origin)]
        report["warmup_minutes"] = args.warmup_minutes

        def replay():
            if dashboard.forecast_worker.forecaster is None:
                QTimer.singleShot(100, replay)
                return
            # Stop wall-clock refresh during explicit historical replay.
            QMetaObject.invokeMethod(dashboard.forecast_worker.timer, "stop", Qt.QueuedConnection)
            for row in rows.to_dict("records"):
                dashboard.receive_sample(dict(row, camera_name=FORECAST_CAMERA))
            QMetaObject.invokeMethod(dashboard.forecast_worker, "refresh", Qt.QueuedConnection, Q_ARG(object, origin))

        QTimer.singleShot(100, replay)
    dashboard.forecast_worker.updated.connect(lambda history, result, camera_name: updates.append(result.alert_level))
    heartbeat = QTimer(app)
    heartbeat.setInterval(50)
    heartbeat.timeout.connect(lambda: heartbeats.append(1))
    heartbeat.start()

    def finish():
        output = ROOT / "reports" / ("ui_wti_live.png" if args.live else ("ui_wti_startup.png" if args.startup else "ui_wti_replay.png"))
        output.parent.mkdir(parents=True, exist_ok=True)
        report.update({
            "window_visible": window.isVisible(), "dashboard_visible": dashboard.isVisible(),
            "actual_points": int(np.isfinite(dashboard.chart.actual_line.get_ydata()).sum()),
            "reference_history_points": int(np.isfinite(dashboard.chart.bootstrap_line.get_ydata()).sum()),
            "forecast_points": len(dashboard.chart.forecast_line.get_ydata()),
            "reference_minutes_in_history": getattr(dashboard, "last_result", None).reference_minutes if hasattr(dashboard, "last_result") else None,
            "is_provisional": getattr(dashboard, "last_result", None).is_provisional if hasattr(dashboard, "last_result") else None,
            "forecast_values_count": len(dashboard.last_result.forecast_values) if hasattr(dashboard, "last_result") else 0,
            "display_span_minutes": float(np.diff(dashboard.chart.ax.get_xlim())[0] * 24 * 60),
            "reference_overlay_visible": dashboard.chart.reference_line.get_visible(),
            "now_centered": bool(np.isclose(sum(dashboard.chart.ax.get_xlim()) / 2, mdates.date2num(dashboard.chart.current_time))),
            "viewer_camera": window.cameraTab.cameraCombo.currentText(),
            "viewer_detection_enabled": window.cameraTab.yoloSwitch.isChecked(),
            "viewer_changes_during_check": viewer_changes,
            "forecast_camera_sources": sorted({sample["camera_name"] for sample in counts}),
            "status": dashboard.status_label.text(), "camera_samples": len(counts),
            "frames_received": len(frames), "ui_heartbeat_ticks": len(heartbeats),
            "worker_updates": updates, "model_path": str(MODEL_PATH),
            "screenshot_saved": window.grab().save(str(output)),
        })
        output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report))
        app.quit()

    QTimer.singleShot(max(1, args.seconds) * 1000, finish)
    code = app.exec()
    window.close()
    expected = report.get("camera_samples", 0) > 0 if args.live else report.get("forecast_values_count") == FORECAST_HORIZON
    return code if expected and report.get("screenshot_saved") else 1


if __name__ == "__main__":
    sys.exit(main())
