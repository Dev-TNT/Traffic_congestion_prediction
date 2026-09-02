"""Threaded bridge between the live traffic camera, YOLO, and the Qt UI."""

import time
import numpy as np

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from utils.app_state import app_state


class CameraWorker(QThread):
    """Periodically fetches traffic frames and emits them to the GUI thread."""

    frame_ready = pyqtSignal(QImage)

    def run(self) -> None:
        """Run the old terminal loop without ever touching Qt widgets directly."""
        # Yolo_detect creates its YOLO model at import time.  Deferring these
        # imports ensures that initialization happens on this worker thread,
        # never while the main window is being created.
        import core.Get_frame as Get_frame
        import core.Yolo_detect as Yolo_detect

        while not self.isInterruptionRequested():
            start_t = time.time()
            frame = Get_frame.get_traffic_image("Camera 1")

            if frame is not None:
                if app_state.show_yolo_frame:
                    (
                        annotated_frame,
                        person_count,
                        car_count,
                        motorbike_count,
                        bus_count,
                        truck_count,
                        total_vehicles,
                    ) = Yolo_detect.image_processing(frame)

                    print(
                        f"Detected - "
                        f"Persons: {person_count}, "
                        f"Cars: {car_count}, "
                        f"Motorbikes: {motorbike_count}, "
                        f"Buses: {bus_count}, "
                        f"Trucks: {truck_count}, "
                        f"Total Vehicles: {total_vehicles}"
                    )
                    display_frame = annotated_frame
                else:
                    display_frame = frame

                qimage = self._bgr_to_qimage(display_frame)
                if qimage is not None:
                    self.frame_ready.emit(qimage)

            elapsed_time = time.time() - start_t
            sleep_time = max(1.0, 10.0 - elapsed_time)
            # The camera API is rate-limited: do not reduce this to a frame-rate sleep.
            self.msleep(int(sleep_time * 1000))

    def stop(self) -> None:
        """Use Qt's native interruption mechanism for graceful shutdown."""
        if self.isRunning():
            self.requestInterruption()
            self.wait()

    @staticmethod
    def _bgr_to_qimage(frame: np.ndarray) -> QImage | None:
        """Convert a BGR NumPy frame to a detached RGB888 QImage."""
        if frame is None:
            return None

        # 1. Chuyển sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        # 2. CHIÊU CUỐI: Ép thẳng thành chuỗi Bytes
        # Cách này diệt 100% lỗi 0xC0000409 và xóa sổ luôn gạch vàng của PyCharm
        image_bytes = rgb_frame.tobytes()

        # 3. Khởi tạo QImage an toàn
        image = QImage(
            image_bytes,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return image.copy()