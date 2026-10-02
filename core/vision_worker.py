"""Threaded bridge between the live traffic camera, YOLO, and the Qt UI."""

import time
from datetime import datetime
import numpy as np

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from utils.app_state import app_state
from core.traffic_features import FORECAST_CAMERA


class CameraWorker(QThread):
    """Periodically fetches traffic frames and emits them to the GUI thread."""

    frame_ready = pyqtSignal(QImage)
    counts_ready = pyqtSignal(object)

    def run(self) -> None:
        """Run capture and YOLO inference outside the GUI thread."""
        # Yolo_detect creates its YOLO model at import time, so import it
        # inside this worker thread to avoid blocking UI startup.
        import core.Get_frame as Get_frame
        import core.Yolo_detect as Yolo_detect

        last_signature = None
        last_detection = None
        while not self.isInterruptionRequested():
            start_t = time.time()

            selected_cam_name = app_state.selected_camera_name
            # Forecast Camera 1 always runs, independently of the viewer controls.
            forecast_frame = Get_frame.get_traffic_image(FORECAST_CAMERA)
            received_at = datetime.now()  # local receipt time, not camera capture time
            if forecast_frame is not None:
                signature = forecast_frame.tobytes()
                if signature != last_signature:
                    last_detection = Yolo_detect.image_processing(forecast_frame)
                    self.counts_ready.emit({
                        "timestamp": received_at, "camera_name": FORECAST_CAMERA,
                        "processing_seconds": (datetime.now() - received_at).total_seconds(),
                        "car": int(last_detection[2]), "motorbike": int(last_detection[3]),
                        "bus": int(last_detection[4]), "truck": int(last_detection[5]),
                        "total": int(last_detection[6]), "WTI": float(last_detection[7]),
                    })
                    last_signature = signature
            frame = forecast_frame if selected_cam_name == FORECAST_CAMERA else Get_frame.get_traffic_image(selected_cam_name)

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
                        weighted_traffic_impact,
                        weighted_traffic_impact_norm,
                    ) = last_detection if selected_cam_name == FORECAST_CAMERA else Yolo_detect.image_processing(frame)

                    app_state.person_count = int(person_count)
                    app_state.car_count = int(car_count)
                    app_state.motorbike_count = int(motorbike_count)
                    app_state.bus_count = int(bus_count)
                    app_state.truck_count = int(truck_count)
                    app_state.total_vehicles = int(total_vehicles)
                    app_state.weighted_traffic_impact = float(weighted_traffic_impact)
                    app_state.weighted_traffic_impact_norm = float(weighted_traffic_impact_norm)
                    display_frame = annotated_frame
                else:
                    display_frame = frame

                qimage = self._bgr_to_qimage(display_frame)
                if qimage is not None:
                    self.frame_ready.emit(qimage)

            elapsed_time = time.time() - start_t
            sleep_time = max(1.0, 10.0 - elapsed_time)
            self.msleep(int(sleep_time * 1000))

    def stop(self) -> None:
        """Request a graceful shutdown using Qt's native interruption API."""
        if self.isRunning():
            self.requestInterruption()
            self.wait()

    @staticmethod
    def _bgr_to_qimage(frame: np.ndarray) -> QImage | None:
        """Convert a BGR OpenCV frame to a detached RGB QImage."""
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        image = QImage(
            rgb_frame.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return image.copy()
