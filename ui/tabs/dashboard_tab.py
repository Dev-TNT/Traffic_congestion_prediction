"""Prediction dashboard and dynamic placeholder traffic visualization."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout
from qfluentwidgets import CaptionLabel, CardWidget, LargeTitleLabel


class DashboardTab(QFrame):
    """Sub-interface that visualizes model predictions and traffic behavior."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("DashboardTab")
        self._init_ui()
        self._plot_placeholder_data()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        prediction_row = QHBoxLayout()
        prediction_row.setSpacing(16)
        self.timeToCongestionCard = self._build_prediction_card(
            "Time until congestion", "-- mins"
        )
        self.durationCard = self._build_prediction_card("Estimated duration", "-- mins")
        prediction_row.addWidget(self.timeToCongestionCard)
        prediction_row.addWidget(self.durationCard)
        main_layout.addLayout(prediction_row)

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_alpha(0.0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)

        chart_card = CardWidget(self)
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(16, 16, 16, 16)
        chart_layout.addWidget(self.canvas)
        main_layout.addWidget(chart_card, stretch=1)

    def _build_prediction_card(self, caption: str, value_text: str) -> CardWidget:
        """Build one large prediction summary card."""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(4)

        caption_label = CaptionLabel(caption, card)
        caption_label.setTextColor("#606060", "#c0c0c0")
        value_label = LargeTitleLabel(value_text, card)
        value_label.setObjectName("valueLabel")
        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        return card

    def _plot_placeholder_data(self) -> None:
        """Generate the supplied NumPy baseline and noisy real-time curve."""
        x = np.linspace(0, 60, 120)
        baseline = 20 + 5 * np.sin(x / 10)
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, 1.5, size=x.shape)
        realtime = baseline + (x / 60) * 25 + noise
        self.update_chart(x, baseline, realtime)

    def update_prediction(self, time_to_congestion_min: float, duration_min: float) -> None:
        """Update the model predictions displayed in the summary cards."""
        time_label = self.timeToCongestionCard.findChild(LargeTitleLabel, "valueLabel")
        duration_label = self.durationCard.findChild(LargeTitleLabel, "valueLabel")

        if time_label:
            time_label.setText(f"{time_to_congestion_min:.0f} mins")
        if duration_label:
            duration_label.setText(f"{duration_min:.0f} mins")

    def update_chart(self, x_data, baseline_data, realtime_data) -> None:
        """Refresh the chart with incoming baseline and observed measurements."""
        self.ax.clear()
        self.ax.plot(
            x_data,
            baseline_data,
            label="Normal Traffic Baseline",
            color="#4C72B0",
            linewidth=2,
        )
        self.ax.plot(
            x_data,
            realtime_data,
            label="Real-time Traffic Behavior",
            color="red",
            linewidth=2,
        )
        self.ax.set_xlabel("Time (minutes)")
        self.ax.set_ylabel("Traffic Density / Vehicle Count")
        self.ax.set_title("Traffic Behavior Over Time")
        self.ax.legend(loc="upper left", fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
