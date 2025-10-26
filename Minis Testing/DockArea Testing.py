from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Area Example with Plot")

        # --- Central Plot ---
        self.plot_widget = pg.PlotWidget(title="Main Plot")
        self.setCentralWidget(self.plot_widget)

        # Add example data
        x = np.linspace(0, 10, 200)
        y = np.sin(x)
        self.plot_widget.plot(x, y, pen='y', name="Sine Wave")

        # --- Left Dock: Controls ---
        left_dock = QDockWidget("Controls", self)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Controls Panel"))
        left_layout.addWidget(QPushButton("Run Simulation"))
        left_layout.addWidget(QPushButton("Update Plot"))
        left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # --- Right Dock: Parameters ---
        right_dock = QDockWidget("Parameters", self)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Parameter 1"))
        right_layout.addWidget(QLabel("Parameter 2"))
        right_layout.addWidget(QPushButton("Apply"))
        right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # --- Bottom Dock: Log ---
        bottom_dock = QDockWidget("Log Output", self)
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.append("System initialized.")
        bottom_dock.setWidget(log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom_dock)

        # --- Dock Options ---
        for dock in [left_dock, right_dock, bottom_dock]:
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        # Set minimum sizes
        self.resize(1200, 700)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
