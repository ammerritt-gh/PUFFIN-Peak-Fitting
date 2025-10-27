# view/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

# -- color palette (change these) --
PLOT_BG = "white"       # plot background
POINT_COLOR = "black"   # scatter points
ERROR_COLOR = "black"   # error bars (can match points)
FIT_COLOR = "red"     # fit line
AXIS_COLOR = "black"    # axis and tick labels
GRID_ALPHA = 0.5

class MainWindow(QMainWindow):
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")
        self.viewmodel = viewmodel

        # --- Central Plot ---
        self.plot_widget = pg.PlotWidget(title="Data and Fit")
        self.setCentralWidget(self.plot_widget)
        self._init_plot()

        # --- Docks ---
        self._init_left_dock()
        self._init_right_dock()
        self._init_bottom_dock()

        for dock in [self.left_dock, self.right_dock, self.bottom_dock]:
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.resize(1400, 800)

    # --------------------------
    # Plot setup
    # --------------------------
    def _init_plot(self):
        # Replace line-plot for data with a scatter + error bars,
        # keep a line plot for the fit.
        # apply background and grid
        self.plot_widget.setBackground(PLOT_BG)
        self.plot_widget.showGrid(x=True, y=True, alpha=GRID_ALPHA)

        # scatter (data points)
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(POINT_COLOR))
        self.plot_widget.addItem(self.scatter)

        # error bars
        self.err_item = pg.ErrorBarItem(pen=pg.mkPen(ERROR_COLOR))
        self.plot_widget.addItem(self.err_item)

        # fit line
        self.fit_curve = self.plot_widget.plot([], [], pen=pg.mkPen(FIT_COLOR, width=2), name="Fit")

        # axis colors (safe: try each axis)
        for ax in ("left", "bottom", "right", "top"):
            try:
                axis = self.plot_widget.getAxis(ax)
                axis.setPen(pg.mkPen(AXIS_COLOR))
                axis.setTextPen(pg.mkPen(AXIS_COLOR))
            except Exception:
                pass

    # --------------------------
    # Docks
    # --------------------------
    def _init_left_dock(self):
        self.left_dock = QDockWidget("Controls", self)
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        load_btn = QPushButton("Load Data")
        save_btn = QPushButton("Save Data")
        fit_btn = QPushButton("Run Fit")
        update_btn = QPushButton("Update Plot")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(fit_btn)
        layout.addWidget(update_btn)
        layout.addStretch(1)

        self.left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # Connect UI → ViewModel
        if self.viewmodel:
            load_btn.clicked.connect(self.viewmodel.load_data)
            save_btn.clicked.connect(self.viewmodel.save_data)
            fit_btn.clicked.connect(self.viewmodel.run_fit)
            update_btn.clicked.connect(self.viewmodel.update_plot)

    def _init_right_dock(self):
        self.right_dock = QDockWidget("Parameters", self)
        param_widget = QWidget()
        layout = QFormLayout(param_widget)

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
            apply_btn.clicked.connect(self._on_apply_clicked)

    def _init_bottom_dock(self):
        self.bottom_dock = QDockWidget("Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("System initialized.")
        self.bottom_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

    # --------------------------
    # View-only public methods
    # --------------------------
    def append_log(self, msg: str):
        self.log_text.append(msg)

    def update_plot_data(self, x, y_data, y_fit=None, y_err=None):
        # Draw scatter points
        if x is None or y_data is None:
            return

        # Ensure numeric numpy arrays are used (prevents list subtraction errors in ErrorBarItem)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)

        self.scatter.setData(x=x_arr, y=y_arr)

        # Draw vertical error bars when provided
        if y_err is not None and len(y_err) == len(y_arr):
            top = np.abs(np.asarray(y_err, dtype=float))
            bottom = top
            self.err_item.setData(x=x_arr, y=y_arr, top=top, bottom=bottom)
        else:
            # clear error bars using numpy arrays (avoid passing Python lists)
            empty = np.array([], dtype=float)
            try:
                self.err_item.setData(x=empty, y=empty, top=empty, bottom=empty)
            except Exception:
                pass

        # Fit line (if present)
        if y_fit is not None:
            yfit_arr = np.asarray(y_fit, dtype=float)
            self.fit_curve.setData(x_arr, yfit_arr)
        else:
            self.fit_curve.clear()

    def _on_apply_clicked(self):
        if not self.viewmodel:
            return
        self.viewmodel.apply_parameters(
            gauss=self.gauss_spin.value(),
            lorentz=self.lorentz_spin.value(),
            temp=self.temp_spin.value()
        )
