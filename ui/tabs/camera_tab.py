"""Live camera interface for camera selection, video, and detection statistics."""

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    IndicatorPosition,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
)

from utils.app_state import app_state


class ImageLabel(QLabel):
    """A scalable video surface that preserves its source image's aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_pixmap: QPixmap | None = None

        self.setObjectName("videoLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background-color: #1e1e1e; border-radius: 8px; color: white; }"
        )
        self.setText("Waiting for camera feed...")

    def set_image(self, image: QImage) -> None:
        """Accept and render a new QImage."""
        if image.isNull():
            return

        self._last_pixmap = QPixmap.fromImage(image)
        self._rescale_and_set_pixmap()

    def resizeEvent(self, event) -> None:
        """Re-render the most recent frame after resizing."""
        super().resizeEvent(event)
        self._rescale_and_set_pixmap()

    def _rescale_and_set_pixmap(self) -> None:
        """Scale the stored frame without distortion."""
        if self._last_pixmap is None:
            return

        scaled = self._last_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class CameraTab(QFrame):
    """Live video feed, camera selector, YOLO toggle, and detection statistics."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CameraTab")

        self._statistics_labels = {}
        self._init_ui()

        self.statisticsTimer = QTimer(self)
        self.statisticsTimer.setInterval(250)
        self.statisticsTimer.timeout.connect(self._refresh_statistics)
        self.statisticsTimer.start()

        self._refresh_statistics()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        self.videoLabel = ImageLabel(self)
        main_layout.addWidget(self.videoLabel, stretch=1)

        statistics_card = CardWidget(self)
        statistics_layout = QVBoxLayout(statistics_card)
        statistics_layout.setContentsMargins(16, 14, 16, 14)
        statistics_layout.setSpacing(10)

        statistics_layout.addWidget(
            SubtitleLabel("Live Detection Statistics", statistics_card)
        )

        statistics_grid = QGridLayout()
        statistics_grid.setHorizontalSpacing(28)
        statistics_grid.setVerticalSpacing(8)

        statistics = (
            ("Persons", "person_count"),
            ("Cars", "car_count"),
            ("Motorbikes", "motorbike_count"),
            ("Buses", "bus_count"),
            ("Trucks", "truck_count"),
            ("Total Vehicles", "total_vehicles"),
        )

        for index, (title, state_attribute) in enumerate(statistics):
            label = StrongBodyLabel(f"{title}: 0", statistics_card)
            self._statistics_labels[state_attribute] = (title, label)
            statistics_grid.addWidget(label, index // 2, index % 2)

        statistics_layout.addLayout(statistics_grid)
        main_layout.addWidget(statistics_card)

        control_card = CardWidget(self)
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(16, 12, 16, 12)

        self.cameraCombo = ComboBox(control_card)
        self.cameraCombo.addItems(list(app_state.camera_ids.keys()))
        self.cameraCombo.setCurrentText(app_state.selected_camera_name)
        self.cameraCombo.currentTextChanged.connect(self._on_camera_changed)

        self.yoloSwitch = SwitchButton(
            control_card,
            indicatorPos=IndicatorPosition.RIGHT,
        )
        self.yoloSwitch.setOnText("YOLO Detection")
        self.yoloSwitch.setOffText("Raw Frame")
        self.yoloSwitch.setChecked(app_state.show_yolo_frame)
        self.yoloSwitch.checkedChanged.connect(self._on_toggle_yolo)

        control_layout.addWidget(BodyLabel("Camera:", control_card))
        control_layout.addWidget(self.cameraCombo)
        control_layout.addSpacing(18)
        control_layout.addWidget(BodyLabel("Display Mode:", control_card))
        control_layout.addStretch(1)
        control_layout.addWidget(self.yoloSwitch)

        main_layout.addWidget(control_card)

    def _on_camera_changed(self, camera_name: str) -> None:
        """Store the selected camera for the next worker fetch cycle."""
        if camera_name in app_state.camera_ids:
            app_state.selected_camera_name = camera_name

    def _on_toggle_yolo(self, checked: bool) -> None:
        """Persist the YOLO/raw display selection."""
        app_state.show_yolo_frame = checked

    def _refresh_statistics(self) -> None:
        """Refresh the stats card from the latest worker-state snapshot."""
        for state_attribute, (title, label) in self._statistics_labels.items():
            value = getattr(app_state, state_attribute) if app_state.show_yolo_frame else "--"
            label.setText(f"{title}: {value}")

    @pyqtSlot(QImage)
    def update_frame(self, image: QImage) -> None:
        """Safely receive a queued frame from CameraWorker."""
        self.videoLabel.set_image(image)
