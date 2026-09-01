"""Minimal object-oriented shared state for the user interface and worker."""


class AppState:
    """Stores UI-controlled values read by background processing workers."""

    def __init__(self) -> None:
        self.show_yolo_frame = True


app_state = AppState()