"""Top-level FluentWindow and its navigation interfaces."""

from qfluentwidgets import FluentIcon, FluentWindow

from ui.tabs.camera_tab import CameraTab
from ui.tabs.dashboard_tab import DashboardTab
from ui.tabs.home_tab import HomeTab


class MainWindow(FluentWindow):
    """Application shell containing home, live camera, and dashboard pages."""

    def __init__(self):
        super().__init__()

        self.homeTab = HomeTab(self)
        self.cameraTab = CameraTab(self)
        self.dashboardTab = DashboardTab(self)

        self.addSubInterface(self.homeTab, FluentIcon.HOME, "Home")
        self.addSubInterface(self.cameraTab, FluentIcon.VIDEO, "Camera")
        self.addSubInterface(self.dashboardTab, FluentIcon.CAR, "Dashboard")

        self.setWindowTitle("Traffic Jam Prediction")
        self.resize(1100, 760)