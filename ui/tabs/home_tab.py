"""Landing and academic project information tab."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    ScrollArea,
    SimpleCardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)


class HomeTab(ScrollArea):
    """A scrollable project overview that remains usable in smaller windows."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self._init_widget()
        self._init_layout()

    def _init_widget(self) -> None:
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("HomeTab")
        self.view.setObjectName("HomeTabView")
        self.setStyleSheet("QScrollArea{background: transparent; border: none}")
        self.view.setStyleSheet("QWidget{background: transparent}")

    def _init_layout(self) -> None:
        self.vBoxLayout.setContentsMargins(36, 36, 36, 36)
        self.vBoxLayout.setSpacing(20)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = TitleLabel("Traffic Jam Prediction", self.view)
        subtitle = SubtitleLabel("Time-to-Congestion Estimation System", self.view)
        subtitle.setTextColor("#606060", "#c0c0c0")
        self.vBoxLayout.addWidget(title)
        self.vBoxLayout.addWidget(subtitle)
        self.vBoxLayout.addSpacing(12)

        card = SimpleCardWidget(self.view)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(10)

        card_layout.addWidget(SubtitleLabel("Project Information", card))
        card_layout.addWidget(StrongBodyLabel("Students:", card))
        card_layout.addWidget(
            BodyLabel(
                "Ho Chi Minh City University of Technology and Education (HCMUTE) "
                "- Electrical and Electronics Engineering",
                card,
            )
        )
        card_layout.addWidget(BodyLabel("• Nguyễn Tuấn Thịnh — MSSV: 25139048", card))
        card_layout.addWidget(BodyLabel("• Nguyễn Đình Tú Bảo — MSSV: 24139005", card))
        card_layout.addSpacing(6)
        card_layout.addWidget(StrongBodyLabel("Instructor:", card))
        card_layout.addWidget(BodyLabel("Nguyễn Mạnh Hùng", card))

        self.vBoxLayout.addWidget(card)
        self.vBoxLayout.addStretch(1)
