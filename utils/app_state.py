"""Minimal shared state for the user interface and vision worker."""


class AppState:
    """Stores UI choices and the latest vehicle-detection results."""

    def __init__(self) -> None:
        self.show_yolo_frame = True

        self.camera_ids = {
            "Camera 1": "56df8198c062921100c143dd",
            "Camera 2": "56df81d8c062921100c143de",
            "Camera 3": "56df8159c062921100c143dc",
        }
        self.selected_camera_name = "Camera 1"

        self.person_count = 0
        self.car_count = 0
        self.motorbike_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.total_vehicles = 0


app_state = AppState()