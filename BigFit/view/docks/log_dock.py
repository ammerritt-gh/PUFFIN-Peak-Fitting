"""
Log dock widget for displaying application log messages.
"""

from PySide6.QtWidgets import QDockWidget, QTextEdit
from PySide6.QtCore import Qt


class LogDock(QDockWidget):
    """Dock widget for displaying log messages."""

    def __init__(self, parent=None):
        """
        Initialize the log dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Log", parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("System initialized.")
        self.setWidget(self.log_text)

    def append_log(self, msg: str):
        """
        Append a message to the log.

        Args:
            msg: Message to append
        """
        try:
            if self.log_text is not None:
                self.log_text.append(msg)
            else:
                print(msg)
        except Exception:
            # avoid raising from logging
            try:
                print(msg)
            except Exception:
                pass
