"""Minimal shared state for the user interface and vision worker."""


class AppState:
    """Stores UI choices and the latest vehicle-detection results."""

    def __init__(self) -> None:
        self.show_yolo_frame = True

        self.camera_ids = {
            "Camera 1": "5d9ddd49766c880017188c94",
            "Camera 2": "5d9ddec9766c880017188c9c",
            "Camera 3": "5b0e1faacddcc80011ceb449",
            "Camera 4": "5d9ddf49766c880017188ca0",
            "Camera 5": "5d9dde1f766c880017188c98"
        }
        self.selected_camera_name = "Camera 1"

        self.person_count = 0
        self.car_count = 0
        self.motorbike_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.total_vehicles = 0


app_state = AppState()