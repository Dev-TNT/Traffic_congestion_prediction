"""Application entry point for the Traffic Jam Prediction desktop UI."""

import sys

from PyQt5.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from core.vision_worker import CameraWorker
from ui.main_window import MainWindow


def main() -> int:
    """Build the UI, start the vision worker, and clean it up on exit."""
    app = QApplication(sys.argv)
    setTheme(Theme.AUTO)

    window = MainWindow()
    camera_worker = CameraWorker(app)
    camera_worker.frame_ready.connect(window.cameraTab.update_frame)
    app.aboutToQuit.connect(camera_worker.stop)

    window.show()
    camera_worker.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())