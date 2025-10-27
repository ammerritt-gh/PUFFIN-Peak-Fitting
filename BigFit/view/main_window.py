# view/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")

        self.viewmodel = viewmodel  # Optional; connect later

        # --- Central Plot ---
        self.plot_widget = pg.PlotWidget(title="Data and Fit")
        self.setCentralWidget(self.plot_widget)
        self._init_plot()

        # --- Left Dock: Controls ---
        self._init_left_dock()

        # --- Right Dock: Parameters ---
        self._init_right_dock()

        # --- Bottom Dock: Log ---
        self._init_bottom_dock()

        # --- Dock Features ---
        for dock in [self.left_dock, self.right_dock, self.bottom_dock]:
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        # --- Window sizing ---
        self.resize(1400, 800)

    # --------------------------
    # Plot setup
    # --------------------------
    def _init_plot(self):
        x = np.linspace(-20, 20, 400)
        y = np.exp(-0.5 * (x / 3) ** 2)  # example data
        self.data_curve = self.plot_widget.plot(x, y, pen='y', name="Data")
        self.fit_curve = self.plot_widget.plot(x, 0 * y, pen='r', name="Fit")

    # --------------------------
    # Docks
    # --------------------------
    def _init_left_dock(self):
        self.left_dock = QDockWidget("Controls", self)
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        load_btn = QPushButton("Load Data")
        fit_btn = QPushButton("Run Fit")
        update_btn = QPushButton("Update Plot")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(load_btn)
        layout.addWidget(fit_btn)
        layout.addWidget(update_btn)
        layout.addStretch(1)

        self.left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # (optional: connect signals later via viewmodel)
        if self.viewmodel:
            load_btn.clicked.connect(self.viewmodel.load_data)
            fit_btn.clicked.connect(self.viewmodel.run_fit)
            update_btn.clicked.connect(self.viewmodel.update_plot)

    def _init_right_dock(self):
        self.right_dock = QDockWidget("Parameters", self)
        param_widget = QWidget()
        layout = QFormLayout(param_widget)

        # Example parameter controls
        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.01, 10.0)
        self.gauss_spin.setValue(1.14)
        layout.addRow("Gaussian FWHM:", self.gauss_spin)

        self.lorentz_spin = QDoubleSpinBox()
        self.lorentz_spin.setRange(0.01, 10.0)
        self.lorentz_spin.setValue(0.28)
        layout.addRow("Lorentzian FWHM:", self.lorentz_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 500.0)
        self.temp_spin.setValue(10.0)
        layout.addRow("Temperature (K):", self.temp_spin)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Voigt", "DHO+Voigt", "Gaussian"])
        layout.addRow("Model:", self.model_combo)

        apply_btn = QPushButton("Apply")
        layout.addRow(apply_btn)

        self.right_dock.setWidget(param_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        if self.viewmodel:
            apply_btn.clicked.connect(self.viewmodel.apply_parameters)

    def _init_bottom_dock(self):
        self.bottom_dock = QDockWidget("Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("System initialized.")
        self.bottom_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

    # --------------------------
    # Public methods for ViewModel
    # --------------------------
    def append_log(self, msg: str):
        self.log_text.append(msg)

    def update_plot_data(self, x, y_data, y_fit=None):
        """Called by ViewModel to refresh display."""
        self.data_curve.setData(x, y_data)
        if y_fit is not None:
            self.fit_curve.setData(x, y_fit)
