"""Live minute-density dashboard hosted in the existing FluentWindow tab."""

import pandas as pd
from PyQt5.QtCore import QThread, QMetaObject, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout
from qfluentwidgets import CaptionLabel, CardWidget, LargeTitleLabel, PushButton, isDarkTheme, qconfig

from core.forecast_worker import ForecastWorker
from core.traffic_features import CONGESTION_THRESHOLD, FORECAST_CAMERA, DISPLAY_WINDOW, MODEL_PATH, TARGET_COLUMN, smooth_observed
from ui.traffic_forecast_chart import TrafficForecastChart
from core.traffic_store import STORE_PATH


STATUS_COLORS = {
    "NORMAL": ("#15803d", "#86efac"), "WATCH": ("#b45309", "#fbbf24"),
    "WARNING": ("#b91c1c", "#fca5a5"),
    "PROVISIONAL": ("#b45309", "#fbbf24"),
    "INSUFFICIENT_DATA": ("#475569", "#cbd5e1"), "MODEL_ERROR": ("#475569", "#cbd5e1"),
}


class DashboardTab(QFrame):
    sample_received = pyqtSignal(object)

    def __init__(self, parent=None, model_path=MODEL_PATH, store_path=STORE_PATH):
        super().__init__(parent=parent)
        self.setObjectName("DashboardTab")
        self._init_ui()
        self._apply_background()
        qconfig.themeChangedFinished.connect(self._apply_background)
        self.forecast_thread = QThread(self)
        self.forecast_worker = ForecastWorker(model_path, store_path)
        self.forecast_worker.moveToThread(self.forecast_thread)
        self.forecast_thread.started.connect(self.forecast_worker.start)
        self.forecast_thread.finished.connect(self.forecast_worker.deleteLater)
        self.sample_received.connect(self.forecast_worker.add_sample)
        self.forecast_worker.updated.connect(self._on_worker_update)
        self.forecast_thread.start()

    def _apply_background(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#202020" if isDarkTheme() else "#f5f7fa"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.summary_labels = {}
        self.summary_captions = {}
        cards = (
            ("current", "WTI hiện tại · TB 3 phút", "-- WTI"),
            ("threshold", "Ngưỡng WTI thử nghiệm", f"{CONGESTION_THRESHOLD:g} WTI"),
            ("trend", "Xu hướng WTI 15 phút", "-- WTI/phút"),
            ("eta", "ETA đầu cửa sổ 3/5", "-- phút"),
        )
        for column, (key, caption, value) in enumerate(cards):
            card = CardWidget(self)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 12, 16, 12)
            caption_label = CaptionLabel(caption, card)
            self.summary_captions[key] = caption_label
            layout.addWidget(caption_label)
            label = LargeTitleLabel(value, card)
            font = label.font()
            font.setPointSize(17)
            label.setFont(font)
            self.summary_labels[key] = label
            layout.addWidget(label)
            grid.addWidget(card, 0, column)
        main_layout.addLayout(grid)
        self.status_label = CaptionLabel("INSUFFICIENT_DATA · Chưa đủ dữ liệu để dự báo.", self)
        self.status_label.setWordWrap(True)
        self.details_label = CaptionLabel("Thời điểm vượt ngưỡng: -- · Cập nhật: --", self)
        self.details_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.details_label)
        controls = QHBoxLayout()
        controls.addWidget(CaptionLabel(f"Nguồn: {FORECAST_CAMERA} · 2 giờ trước / 2 giờ sau · dự báo 1 giờ", self))
        controls.addStretch(1)
        self.reference_button = PushButton("Hiện hành vi thường nhật", self)
        self.reference_button.setCheckable(True)
        self.reference_button.toggled.connect(self._toggle_reference)
        controls.addWidget(self.reference_button)
        main_layout.addLayout(controls)
        chart_card = CardWidget(self)
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        self.chart = TrafficForecastChart(chart_card)
        chart_layout.addWidget(self.chart)
        main_layout.addWidget(chart_card, stretch=1)
        for label in self.findChildren(CaptionLabel) + self.findChildren(LargeTitleLabel):
            label.setTextColor("#172033", "#f1f5f9")

    @pyqtSlot(object)
    def receive_sample(self, sample):
        self.sample_received.emit(sample)

    @pyqtSlot(bool)
    def _toggle_reference(self, checked):
        self.chart.set_reference_visible(checked)
        self.reference_button.setText("Ẩn hành vi thường nhật" if checked else "Hiện hành vi thường nhật")

    @pyqtSlot(object, object, str)
    def _on_worker_update(self, history, result, camera_name):
        if camera_name == FORECAST_CAMERA:
            self.update_forecast(history, result)

    @pyqtSlot(object, object)
    def update_forecast(self, history, result):
        self.last_result = result
        actual = history.tail(DISPLAY_WINDOW + 1)
        times = actual.index if not actual.empty else []
        smoothed = smooth_observed(actual) if not actual.empty else pd.Series(dtype=float)
        values = smoothed.to_numpy()
        reference_mask = actual.is_reference.to_numpy() if "is_reference" in actual else None
        self.chart.update_actual(times, values, reference_mask)
        reference = history.attrs.get("daily_reference")
        self.chart.update_reference(reference.index if reference is not None else [], reference[TARGET_COLUMN].to_numpy() if reference is not None else [])
        current = smoothed.iloc[-1] if not actual.empty else float("nan")
        self.chart.update_forecast(result.forecast_times, result.forecast_values, result.threshold, result.crossing_time, result.is_provisional, current)
        self.chart.show_status(result.status_message)
        current_is_reference = not actual.empty and "is_reference" in actual and actual.is_reference.iloc[-1]
        self.summary_captions["current"].setText("WTI hiện tại · tham chiếu" if current_is_reference else "WTI Camera 1 · TB 3 phút")
        self.summary_labels["current"].setText(f"{current:.1f} WTI" if pd.notna(current) else "-- WTI")
        self.summary_labels["threshold"].setText(f"{result.threshold:g} WTI")
        trend = None
        if len(actual) >= 16:
            recent = smoothed.iloc[-16:]
            same_source = "is_reference" not in actual or actual.is_reference.iloc[-16:].nunique() == 1
            if recent.notna().all() and same_source:
                trend = (recent.iloc[-1] - recent.iloc[0]) / 15
        self.summary_labels["trend"].setText(f"{trend:+.2f} WTI/phút" if trend is not None else "-- WTI/phút")
        eta_text = "-- phút" if not result.is_ready else (
            "Chưa thấy vượt" if result.eta_minutes is None else (
                "Đang vượt ngưỡng" if result.eta_minutes == 0 else f"~{result.eta_minutes} phút")
        )
        self.summary_labels["eta"].setText(eta_text)
        self.status_label.setText(f"{result.alert_level} · {result.status_message}")
        self.status_label.setTextColor(*STATUS_COLORS[result.alert_level])
        crossing = result.crossing_time.strftime("%d/%m %H:%M") if result.crossing_time is not None else "--"
        updated = actual.index[-1].strftime("%d/%m %H:%M") if not actual.empty else "--"
        age = history.attrs.get("sample_age_seconds")
        freshness = f" · Mẫu gần nhất: {age:.0f} giây trước" if age is not None else ""
        self.details_label.setText(f"Đầu cửa sổ vượt ngưỡng: {crossing} · Cập nhật: {updated}{freshness}\n{result.quality_message}")

    def stop(self):
        if self.forecast_thread.isRunning():
            QMetaObject.invokeMethod(self.forecast_worker, "stop", Qt.BlockingQueuedConnection)
        self.forecast_thread.quit()
        self.forecast_thread.wait()
