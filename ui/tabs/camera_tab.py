"""Live camera interface for raw and YOLO-annotated frames."""

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout
from qfluentwidgets import BodyLabel, CardWidget, IndicatorPosition, SwitchButton

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
        """Accept an incoming QImage and render it at the current label size."""
        if image.isNull():
            return

        self._last_pixmap = QPixmap.fromImage(image)
        self._rescale_and_set_pixmap()

    def resizeEvent(self, event) -> None:
        """Re-render the latest image when the display surface changes size."""
        super().resizeEvent(event)
        self._rescale_and_set_pixmap()

    def _rescale_and_set_pixmap(self) -> None:
        """Scale the stored frame without distorting it."""
        if self._last_pixmap is None:
            return

        scaled = self._last_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class CameraTab(QFrame):
    """Sub-interface showing the live video feed and YOLO/raw toggle."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CameraTab")
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        self.videoLabel = ImageLabel(self)
        main_layout.addWidget(self.videoLabel, stretch=1)

        control_card = CardWidget(self)
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(16, 12, 16, 12)

        self.yoloSwitch = SwitchButton(control_card, indicatorPos=IndicatorPosition.RIGHT)
        self.yoloSwitch.setOnText("YOLO Detection")
        self.yoloSwitch.setOffText("Raw Frame")
        self.yoloSwitch.setChecked(app_state.show_yolo_frame)
        self.yoloSwitch.checkedChanged.connect(self._on_toggle_yolo)

        control_layout.addWidget(BodyLabel("Display Mode:", control_card))
        control_layout.addStretch(1)
        control_layout.addWidget(self.yoloSwitch)
        main_layout.addWidget(control_card)

    def _on_toggle_yolo(self, checked: bool) -> None:
        """Persist the switch choice so the camera worker can read it."""
        app_state.show_yolo_frame = checked

    @pyqtSlot(QImage)
    def update_frame(self, image: QImage) -> None:
        """Safely receive a worker frame through Qt's queued signal delivery."""
        self.videoLabel.set_image(image)
