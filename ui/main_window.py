"""Top-level FluentWindow and its navigation interfaces."""

from qfluentwidgets import FluentIcon, FluentWindow, isDarkTheme, qconfig
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QLabel

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
        for label in self.findChildren(QLabel):
            if hasattr(label, "setTextColor"):
                label.setTextColor("#172033", "#f1f5f9")
        self._apply_text_contrast()
        qconfig.themeChangedFinished.connect(self._apply_text_contrast)

        self.addSubInterface(self.homeTab, FluentIcon.HOME, "Home")
        self.addSubInterface(self.cameraTab, FluentIcon.VIDEO, "Camera")
        self.addSubInterface(self.dashboardTab, FluentIcon.CAR, "Dashboard")

        self.setWindowTitle("Traffic Jam Prediction")
        self.resize(1100, 760)

    def _apply_text_contrast(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#202020" if isDarkTheme() else "#f5f7fa"))
        palette.setColor(QPalette.WindowText, QColor("#f1f5f9" if isDarkTheme() else "#172033"))
        self.setPalette(palette)
