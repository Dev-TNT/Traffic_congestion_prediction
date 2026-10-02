"""Reusable PyQt5 chart; its Figure and artists are created once."""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qfluentwidgets import isDarkTheme, qconfig

from core.traffic_features import CONGESTION_THRESHOLD, DISPLAY_WINDOW


class TrafficForecastChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.actual_line, = self.ax.plot([], [], color="#60a5fa", linewidth=2, marker=".", markersize=3, label="Camera 1 · WTI TB 3 phút")
        self.bootstrap_line, = self.ax.plot([], [], color="#94a3b8", linestyle=":", linewidth=2, label="Lịch sử tham chiếu")
        self.forecast_line, = self.ax.plot([], [], color="#f59e0b", linewidth=2, label="Dự báo WTI 60 phút")
        self.reference_line, = self.ax.plot([], [], color="#8b5cf6", linestyle="--", linewidth=1.7, label="Thường nhật · data1")
        self.reference_line.set_visible(False)
        self.threshold_line = self.ax.axhline(CONGESTION_THRESHOLD, color="#dc2626", linestyle="--", label="Ngưỡng thử nghiệm")
        now = datetime.now()
        self.now_line = self.ax.axvline(now, color="#64748b", linestyle=":", label="Hiện tại")
        self.crossing_line = self.ax.axvline(now, color="#dc2626", linestyle="-.", label="Đầu cửa sổ 3/5")
        self.crossing_line.set_visible(False)
        self.ax.set_xlim(now - timedelta(minutes=DISPLAY_WINDOW), now + timedelta(minutes=DISPLAY_WINDOW))
        self.ax.set_ylim(0, CONGESTION_THRESHOLD * 1.3)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.set_xlabel("Giờ Việt Nam · phút hoàn tất")
        self.ax.set_ylabel("WTI · điểm trọng số phương tiện/ảnh")
        self.ax.set_title("Dự báo WTI 60 phút · trung bình trượt 3 phút", loc="left", fontsize=11)
        self.ax.grid(True, alpha=0.18)
        self.ax.legend(loc="upper left", fontsize=7, ncol=3)
        self.status_label = QLabel("Chưa đủ dữ liệu để dự báo.", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.status_label)
        self.current_time = now
        self._provisional = False
        qconfig.themeChanged.connect(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self):
        dark = isDarkTheme()
        background, foreground = ("#202832", "#f1f5f9") if dark else ("#ffffff", "#172033")
        self.figure.set_facecolor(background)
        self.ax.set_facecolor(background)
        self.ax.tick_params(colors=foreground)
        for item in (self.ax.xaxis.label, self.ax.yaxis.label, self.ax.title):
            item.set_color(foreground)
        self.ax.set_title(self.ax.get_title(loc="left"), loc="left", color=foreground)
        for spine in self.ax.spines.values():
            spine.set_color("#64748b" if dark else "#94a3b8")
        legend = self.ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor("#64748b")
            for label in legend.get_texts():
                label.set_color(foreground)
        self.status_label.setStyleSheet(f"color: {foreground}; font-size: 12px;")
        self.canvas.draw_idle()

    def update_actual(self, times, values, reference_mask=None):
        values = np.asarray(values, dtype=float)
        mask = np.zeros(len(values), dtype=bool) if reference_mask is None else np.asarray(reference_mask, dtype=bool)
        self.actual_line.set_data(times, np.where(mask, np.nan, values))
        self.bootstrap_line.set_data(times, np.where(mask, values, np.nan))
        self.current_time = times[-1] if len(times) else datetime.now()
        self.now_line.set_xdata([self.current_time, self.current_time])
        self._redraw()

    def update_reference(self, times, values):
        self.reference_line.set_data(times, values)
        self._redraw()

    def set_reference_visible(self, visible):
        self.reference_line.set_visible(visible)
        self._redraw()

    def update_forecast(self, forecast_times, forecast_values, threshold, crossing_time, is_provisional=False, anchor_value=None):
        times, values = list(forecast_times), list(forecast_values)
        if times and anchor_value is not None and np.isfinite(anchor_value):
            times.insert(0, self.current_time)
            values.insert(0, anchor_value)
        self.forecast_line.set_data(times, values)
        if times and is_provisional != self._provisional:
            self._provisional = is_provisional
            self.forecast_line.set_linestyle("--" if is_provisional else "-")
            self.forecast_line.set_label("WTI tạm thời 60 phút · TB 3p" if is_provisional else "Model WTI 60 phút · TB 3p")
            self.ax.legend(loc="upper left", fontsize=7, ncol=3)
            self._apply_theme()
        self.threshold_line.set_ydata([threshold, threshold])
        self.crossing_line.set_visible(crossing_time is not None)
        if crossing_time is not None:
            self.crossing_line.set_xdata([crossing_time, crossing_time])
        self._redraw()

    def show_status(self, message):
        self.status_label.setText(message)

    def _redraw(self):
        self.ax.set_xlim(self.current_time - timedelta(minutes=DISPLAY_WINDOW), self.current_time + timedelta(minutes=DISPLAY_WINDOW))
        values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in (
            self.actual_line, self.bootstrap_line, self.forecast_line, self.threshold_line,
        )] + ([np.asarray(self.reference_line.get_ydata(), dtype=float)] if self.reference_line.get_visible() else []))
        finite = values[np.isfinite(values)]
        self.ax.set_ylim(0, max(10, float(finite.max()) * 1.2) if len(finite) else 50)
        self.canvas.draw_idle()
