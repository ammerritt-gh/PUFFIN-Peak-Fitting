# view/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox,
    QDialog, QLineEdit, QDialogButtonBox, QHBoxLayout, QFileDialog,
    QScrollArea, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from view.view_box import CustomViewBox
from view.input_handler import InputHandler

# -- color palette (change these) --
PLOT_BG = "white"       # plot background
POINT_COLOR = "black"   # scatter points
ERROR_COLOR = "black"   # error bars (can match points)
FIT_COLOR = "purple"     # fit line
AXIS_COLOR = "black"    # axis and tick labels
GRID_ALPHA = 0.5

class MainWindow(QMainWindow):
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")
        self.viewmodel = viewmodel
        self.param_widgets = {}   # name -> widget
        self.curves = {}  # curve_id -> PlotDataItem
        self.selected_curve_id = None

        # --- Central Plot ---
        self.plot_widget = pg.PlotWidget(title="Data and Fit")
        self.setCentralWidget(self.plot_widget)
        self._init_plot()

        # --- Docks ---
        self._init_left_dock()
        # create the bottom (log) dock before the right dock so logging is available
        self._init_bottom_dock()
        self._init_right_dock()

        for dock in [self.left_dock, self.right_dock, self.bottom_dock]:
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.resize(1400, 800)

    # --------------------------
    # Plot setup
    # --------------------------
    def _init_plot(self):
        # Use custom ViewBox for interactive behavior
        self.viewbox = CustomViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.viewbox, title="Data and Fit")
        self.setCentralWidget(self.plot_widget)

        # optional InputHandler for global mouse/key binding (pass viewbox so it can connect signals)
        self.input_handler = InputHandler(self.viewbox, self.viewmodel)
        self.plot_widget.installEventFilter(self.input_handler)
        # Also install event filter on the ViewBox so we can intercept wheel events
        # (ViewBox handles zoom by default). This allows the InputHandler to
        # intercept wheel events when a curve is selected and use them for
        # parameter updates instead of zooming.
        try:
            self.viewbox.installEventFilter(self.input_handler)
            # keep a backref so the ViewBox can delegate wheel handling when needed
            try:
                setattr(self.viewbox, "_input_handler", self.input_handler)
            except Exception:
                pass
        except Exception:
            pass

        # connect ViewBox interaction signals → ViewModel
        if self.viewmodel:
            self.viewbox.peakSelected.connect(self._on_peak_selected)
            self.viewbox.peakDeselected.connect(lambda: self.viewmodel.set_selected_curve(None))
            self.viewbox.peakMoved.connect(self._on_peak_moved)
            self.viewbox.excludePointClicked.connect(self._on_exclude_point)
            self.viewbox.excludeBoxDrawn.connect(self._on_exclude_box)

        # connect curve selection changes
        # Let the InputHandler know when the selected curve changes (no notify back)
        try:
            self.viewmodel.curve_selection_changed.connect(lambda cid: self.input_handler.set_selected_curve(cid, notify_vm=False))
        except Exception:
            pass
        self.viewmodel.curve_selection_changed.connect(self._on_curve_selected)


        # --- visual setup ---
        self.plot_widget.setBackground(PLOT_BG)
        self.plot_widget.showGrid(x=True, y=True, alpha=GRID_ALPHA)

        # scatter (data points)
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(POINT_COLOR))
        self.plot_widget.addItem(self.scatter)
        # excluded points overlay (small grey 'x' markers)
        try:
            self.excluded_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen('gray'), brush=None, symbol='x')
            self.plot_widget.addItem(self.excluded_scatter)
        except Exception:
            self.excluded_scatter = None

        # error bars
        self.err_item = pg.ErrorBarItem(pen=pg.mkPen(ERROR_COLOR))
        self.plot_widget.addItem(self.err_item)

        # fit line
        self.fit_curve = self.plot_widget.plot([], [], pen=pg.mkPen(FIT_COLOR, width=2), name="Fit")

        # axis colors
        for ax in ("left", "bottom", "right", "top"):
            try:
                axis = self.plot_widget.getAxis(ax)
                axis.setPen(pg.mkPen(AXIS_COLOR))
                axis.setTextPen(pg.mkPen(AXIS_COLOR))
            except Exception:
                pass


    # --------------------------
    # Plot interaction callbacks
    # --------------------------
    def _on_peak_selected(self, x, y):
        # Only forward peak selection to the ViewModel when a curve is selected
        try:
            if hasattr(self, "input_handler") and getattr(self.input_handler, "selected_curve_id", None) is None:
                # ignore peak selection unless a curve is selected
                self.append_log("Peak clicked but no curve selected — ignored.")
                return
        except Exception:
            pass

        if self.viewmodel and hasattr(self.viewmodel, "on_peak_selected"):
            self.viewmodel.on_peak_selected(x, y)
        self.append_log(f"Selected peak near ({x:.3f}, {y:.3f})")

    def _on_peak_moved(self, info):
        # Only forward peak movement when a curve is selected
        try:
            if hasattr(self, "input_handler") and getattr(self.input_handler, "selected_curve_id", None) is None:
                self.append_log("Peak moved but no curve selected — ignored.")
                return
        except Exception:
            pass

        if self.viewmodel and hasattr(self.viewmodel, "on_peak_moved"):
            self.viewmodel.on_peak_moved(info)
        self.append_log(f"Moved peak → center={info.get('center', 0):.3f}")

    def _on_exclude_point(self, x, y):
        if self.viewmodel and hasattr(self.viewmodel, "on_exclude_point"):
            self.viewmodel.on_exclude_point(x, y)
        self.append_log(f"Toggled exclusion at ({x:.3f}, {y:.3f})")

    def _on_exclude_box(self, x0, y0, x1, y1):
        if self.viewmodel and hasattr(self.viewmodel, "on_exclude_box"):
            self.viewmodel.on_exclude_box(x0, y0, x1, y1)
        self.append_log(f"Box exclusion from ({x0:.3f}, {y0:.3f}) → ({x1:.3f}, {y1:.3f})")

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
        clear_btn = QPushButton("Clear Plot")
        config_btn = QPushButton("Edit Config")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(fit_btn)
        layout.addWidget(config_btn)
        layout.addWidget(update_btn)
        layout.addWidget(clear_btn)
        # Exclude toggle (click to enable box/point exclusion)
        exclude_btn = QPushButton("Exclude")
        exclude_btn.setCheckable(True)
        layout.addWidget(exclude_btn)
        # keep a reference for external updates (e.g. hotkey toggles)
        self.exclude_btn = exclude_btn
        # wire the button to the viewbox so clicking updates exclude mode
        try:
            if hasattr(self, "viewbox") and self.viewbox is not None:
                exclude_btn.toggled.connect(self.viewbox.set_exclude_mode)
                # Keep the button state synced when the viewbox mode is changed
                if hasattr(self.viewbox, "excludeModeChanged"):
                    try:
                        self.viewbox.excludeModeChanged.connect(exclude_btn.setChecked)
                    except Exception:
                        pass
        except Exception:
            pass
        layout.addStretch(1)

        self.left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # Connect UI → ViewModel
        if self.viewmodel:
            load_btn.clicked.connect(self.viewmodel.load_data)
            save_btn.clicked.connect(self.viewmodel.save_data)
            fit_btn.clicked.connect(self.viewmodel.run_fit)
            update_btn.clicked.connect(self.viewmodel.update_plot)
            # Clear plot button resets to synthetic initial data and clears saved last file
            try:
                clear_btn.clicked.connect(getattr(self.viewmodel, "clear_plot", lambda: None))
            except Exception:
                pass
            config_btn.clicked.connect(self._on_edit_config_clicked)
            # Exclude mode button toggles the CustomViewBox exclude_mode
            def _on_exclude_toggled(checked):
                try:
                    self.viewbox.set_exclude_mode(bool(checked))
                    # ensure input handler knows selection state
                    try:
                        self.input_handler.selected_curve_id = self.input_handler.selected_curve_id
                    except Exception:
                        pass
                    if checked:
                        self.append_log("Exclude mode enabled.")
                    else:
                        self.append_log("Exclude mode disabled.")
                except Exception as e:
                    self.append_log(f"Failed to toggle exclude mode: {e}")

            exclude_btn.toggled.connect(_on_exclude_toggled)

    def _init_right_dock(self):
        # Replaced static parameter controls with a dynamic, scrollable form.
        self.right_dock = QDockWidget("Parameters", self)
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # Model selector placed at the top of the parameters panel
        self.model_selector = QComboBox()
        # Provide common model names; viewmodel.get_parameters / get_model_spec will accept these.
        self.model_selector.addItems(["Voigt", "Gaussian"])
        vlayout.addWidget(QLabel("Model:"))
        vlayout.addWidget(self.model_selector)

        # set initial selection from viewmodel/state if available
        try:
            if self.viewmodel:
                initial = getattr(self.viewmodel.state, "model_name", None)
                if initial:
                    idx = self.model_selector.findText(str(initial), Qt.MatchFixedString)
                    if idx >= 0:
                        self.model_selector.setCurrentIndex(idx)
        except Exception:
            pass

        # when user changes model, instruct viewmodel and refresh the parameter panel
        def _on_model_selected(idx):
            name = self.model_selector.currentText()
            try:
                if self.viewmodel and hasattr(self.viewmodel, "set_model"):
                    self.viewmodel.set_model(name)
                # refresh UI parameters for the selected model
                self._refresh_parameters()
            except Exception as e:
                self.append_log(f"Failed to switch model: {e}")

        self.model_selector.currentIndexChanged.connect(_on_model_selected)

        # Scrollable area to hold the form (so many parameters fit)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)

        # initial empty form widget (will be replaced by _populate_parameters)
        self.param_form_widget = QWidget()
        self.param_form = QFormLayout(self.param_form_widget)
        self.param_scroll.setWidget(self.param_form_widget)

        vlayout.addWidget(self.param_scroll)

        # Apply + Refresh buttons
        btn_h = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        refresh_btn = QPushButton("Refresh")
        btn_h.addWidget(apply_btn)
        btn_h.addWidget(refresh_btn)
        vlayout.addLayout(btn_h)

        self.right_dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        # Make the parameters dock wider by default so controls and hints are visible
        try:
            # A minimum width allows the user to resize smaller/larger while
            # providing a comfortable default layout on startup.
            self.right_dock.setMinimumWidth(360)
        except Exception:
            pass

        # Connect to ViewModel (if present)
        if self.viewmodel:
            apply_btn.clicked.connect(self._on_apply_clicked)
            refresh_btn.clicked.connect(self._refresh_parameters)
            # initial populate
            self._refresh_parameters()

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
        # Safely append to the log widget if present; otherwise fall back to stdout.
        try:
            if hasattr(self, "log_text") and self.log_text is not None:
                self.log_text.append(msg)
            else:
                print(msg)
        except Exception:
            # avoid raising from logging
            try:
                print(msg)
            except Exception:
                pass

    def update_plot_data(self, x, y_data, y_fit=None, y_err=None):
        # Draw scatter points
        if x is None or y_data is None:
            return

        # Ensure numeric numpy arrays are used (prevents list subtraction errors in ErrorBarItem)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)

        # If the ViewModel exposes an exclusion mask, split included/excluded points
        excl_mask = None
        try:
            if self.viewmodel is not None and hasattr(self.viewmodel, "state") and hasattr(self.viewmodel.state, "excluded"):
                excl_mask = np.asarray(self.viewmodel.state.excluded, dtype=bool)
                # ensure mask length matches
                if len(excl_mask) != len(x_arr):
                    # mismatch — ignore mask and log for debugging
                    try:
                        self.append_log(f"Exclusion mask length {len(excl_mask)} != x length {len(x_arr)} — ignoring exclusions")
                    except Exception:
                        pass
                    excl_mask = None
        except Exception:
            excl_mask = None
        # normalize incoming error array as numpy array when present
        y_err_arr = None
        if y_err is not None:
            try:
                y_err_arr = np.asarray(y_err, dtype=float)
            except Exception:
                y_err_arr = None

        if excl_mask is None:
            # no exclusions — show all points in primary scatter
            self.scatter.setData(x=x_arr, y=y_arr)
            # clear excluded overlay
            if self.excluded_scatter is not None:
                try:
                    self.excluded_scatter.setData(x=np.array([], dtype=float), y=np.array([], dtype=float))
                except Exception:
                    pass
            # Draw vertical error bars when provided (all points)
            if y_err_arr is not None and len(y_err_arr) == len(y_arr):
                top = np.abs(y_err_arr)
                bottom = top
                self.err_item.setData(x=x_arr, y=y_arr, top=top, bottom=bottom)
            else:
                # clear error bars using numpy arrays (avoid passing Python lists)
                empty = np.array([], dtype=float)
                try:
                    self.err_item.setData(x=empty, y=empty, top=empty, bottom=empty)
                except Exception:
                    pass
        else:
            incl = ~excl_mask
            try:
                x_in = x_arr[incl]
                y_in = y_arr[incl]
            except Exception:
                x_in = np.array([], dtype=float)
                y_in = np.array([], dtype=float)
            try:
                x_ex = x_arr[excl_mask]
                y_ex = y_arr[excl_mask]
            except Exception:
                x_ex = np.array([], dtype=float)
                y_ex = np.array([], dtype=float)

            self.scatter.setData(x=x_in, y=y_in)
            if self.excluded_scatter is not None:
                try:
                    self.excluded_scatter.setData(x=x_ex, y=y_ex)
                except Exception:
                    pass

            # Error bars only for included points
            if y_err_arr is not None:
                try:
                    # if full-length, index it by included mask; if it already matches included length, use directly
                    if len(y_err_arr) == len(y_arr):
                        top = np.abs(y_err_arr[incl])
                        bottom = top
                        self.err_item.setData(x=x_in, y=y_in, top=top, bottom=bottom)
                    elif len(y_err_arr) == len(x_in):
                        top = np.abs(y_err_arr)
                        bottom = top
                        self.err_item.setData(x=x_in, y=y_in, top=top, bottom=bottom)
                    else:
                        empty = np.array([], dtype=float)
                        try:
                            self.err_item.setData(x=empty, y=empty, top=empty, bottom=empty)
                        except Exception:
                            pass
                except Exception:
                    pass

        # Fit line (if present)
        if y_fit is not None:
            yfit_arr = np.asarray(y_fit, dtype=float)
            if "fit" not in self.curves:
                curve = self.plot_widget.plot(x_arr, yfit_arr, pen=pg.mkPen(FIT_COLOR, width=2))
                self.curves["fit"] = curve
                curve.curve_id = "fit"
                # Enable clicking on curve
                curve.scene().sigMouseClicked.connect(lambda ev, cid="fit": self._on_curve_clicked(ev, cid))
            else:
                self.curves["fit"].setData(x_arr, yfit_arr)
        else:
            # remove fit curve if exists
            if "fit" in self.curves:
                self.plot_widget.removeItem(self.curves["fit"])
                del self.curves["fit"]


    def _on_apply_clicked(self):
        if not self.viewmodel:
            return

        # Collect values dynamically from widgets into a dict
        params = {}
        for name, widget in self.param_widgets.items():
            try:
                if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                    params[name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    params[name] = widget.text()
                else:
                    # fallback: try to read "value" attribute if a custom widget
                    params[name] = getattr(widget, "value", lambda: None)()
            except Exception:
                params[name] = None

        # Send params dict to ViewModel; ViewModel handles validation/usage
        try:
            # ViewModel.apply_parameters should accept a dict of {name: value}
            self.viewmodel.apply_parameters(params)
            self.append_log("Parameters applied.")
        except Exception as e:
            self.append_log(f"Failed to apply parameters: {e}")

    # --------------------------
    # Dynamic parameter helpers
    # --------------------------
    def _refresh_parameters(self):
        """Ask the ViewModel for parameter specs and rebuild the form."""
        if not self.viewmodel:
            return
        try:
            specs = getattr(self.viewmodel, "get_parameters", lambda: {})()
            if specs is None:
                specs = {}
            # allow list-like or dict-like returns
            if isinstance(specs, list):
                # list of (name, spec) or list of names -> convert to dict
                converted = {}
                for item in specs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        converted[item[0]] = item[1]
                    elif isinstance(item, str):
                        converted[item] = None
                specs = converted
            elif not isinstance(specs, dict):
                # unknown shape — nothing to do
                specs = {}
            self._populate_parameters(specs)
            self.append_log("Parameter panel refreshed.")
        except Exception as e:
            self.append_log(f"Failed to refresh parameters: {e}")

    def _populate_parameters(self, specs: dict):
        """Populate the parameter form from a specs dict.

        Each specs entry may be:
          name: value                      # infer type from value
          name: { 'value': v, 'type': 'float'|'int'|'str'|'bool'|'choice',
                  'min': ..., 'max': ..., 'choices': [...], 'decimals': ..., 'step': ... }
        Note: 'decimals' controls the number of decimal places shown by the
        QDoubleSpinBox (display precision). 'step' controls the single-step
        increment used when the user clicks the up/down arrows.
        """
        # Build a fresh form widget and replace the scroll area's widget
        new_widget = QWidget()
        new_form = QFormLayout(new_widget)
        new_param_widgets = {}

        for name, spec in specs.items():
            # Normalize spec into dict with at least 'value' and 'type'
            if isinstance(spec, dict):
                val = spec.get("value")
                ptype = spec.get("type", None)
            else:
                val = spec
                ptype = None

            # Infer type if not provided
            if ptype is None:
                if isinstance(val, bool):
                    ptype = "bool"
                elif isinstance(val, int) and not isinstance(val, bool):
                    ptype = "int"
                elif isinstance(val, float):
                    ptype = "float"
                elif isinstance(val, (list, tuple)):
                    ptype = "choice"
                else:
                    ptype = "str"

            widget = None
            try:
                if ptype == "float":
                    w = QDoubleSpinBox()
                    # Apply provided min/max if available, otherwise use safe large bounds
                    w.setRange(spec.get("min", -1e9), spec.get("max", 1e9) if isinstance(spec, dict) else 1e9)
                    # 'decimals' controls how many decimal places the spinbox displays.
                    w.setDecimals(spec.get("decimals", 6) if isinstance(spec, dict) else 6)
                    # 'step' controls the increment when the user clicks arrows or presses keys
                    w.setSingleStep(spec.get("step", 0.1) if isinstance(spec, dict) else 0.1)
                    if val is not None:
                        w.setValue(float(val))
                    widget = w

                elif ptype == "int":
                    w = QSpinBox()
                    w.setRange(int(spec.get("min", -2147483648)), int(spec.get("max", 2147483647)))
                    w.setSingleStep(int(spec.get("step", 1)))
                    if val is not None:
                        w.setValue(int(val))
                    widget = w

                elif ptype == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(val))
                    widget = w

                elif ptype == "choice":
                    w = QComboBox()
                    choices = []
                    if isinstance(spec, dict) and "choices" in spec:
                        choices = spec.get("choices") or []
                    elif isinstance(val, (list, tuple)):
                        choices = list(val)
                    for c in choices:
                        w.addItem(str(c))
                    # set current
                    cur = spec.get("value") if isinstance(spec, dict) else (val[0] if val else "")
                    if cur is not None:
                        idx = w.findText(str(cur))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                    widget = w

                else:  # str and fallback
                    w = QLineEdit()
                    if val is not None:
                        w.setText(str(val))
                    widget = w
            except Exception:
                # on any error, fallback to a simple line edit
                w = QLineEdit()
                if val is not None:
                    w.setText(str(val))
                widget = w

            # Keep the label text as the parameter name. If the spec provides an
            # interactive control hint, expose it both as a tooltip and as a
            # small visible hint label next to the control so users can discover
            # what mouse/keyboard actions affect the parameter.
            hint = ""
            try:
                if isinstance(spec, dict) and spec.get("control"):
                    ctrl = spec.get("control")
                    action = ctrl.get("action") or ""
                    mods = "+".join(ctrl.get("modifiers", [])) if ctrl.get("modifiers") else ""
                    hint = f"{action}" + (f"+{mods}" if mods else "")
                    try:
                        widget.setToolTip(f"Interactive: {hint}")
                    except Exception:
                        pass
            except Exception:
                pass

            # Container so we can show the widget and a right-hand hint label
            container_h = QWidget()
            hbox = QHBoxLayout(container_h)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.addWidget(widget)
            if hint:
                hint_label = QLabel(f"({hint})")
                # subtle visual style so it's unobtrusive but readable
                hint_label.setStyleSheet("color: gray; font-size: 11px;")
                hint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                hint_label.setToolTip(f"Interactive control: {hint}")
                hbox.addWidget(hint_label)
            else:
                # keep layout stable when no hint exists
                hbox.addWidget(QLabel(""))

            new_form.addRow(name + ":", container_h)
            new_param_widgets[name] = widget

        # Replace the widget shown in the scroll area (frees old widgets)
        self.param_scroll.takeWidget()
        self.param_scroll.setWidget(new_widget)
        self.param_form_widget = new_widget
        self.param_form = new_form
        # Build control map for InputHandler based on specs' optional 'control' entries.
        control_map = {}
        try:
            for name, spec in specs.items():
                try:
                    if isinstance(spec, dict) and "control" in spec:
                        ctrl = spec.get("control") or {}
                        action = ctrl.get("action")
                        mods = tuple(sorted(ctrl.get("modifiers", []))) if ctrl.get("modifiers") is not None else tuple()
                        sensitivity = float(ctrl.get("sensitivity", 1.0))
                        key = (action, mods)
                        control_map.setdefault(key, []).append({"name": name, "sensitivity": sensitivity})
                except Exception:
                    continue
        except Exception:
            control_map = {}

        # Provide control map to input handler if available
        try:
            if hasattr(self, "input_handler") and self.input_handler is not None:
                self.input_handler.set_control_map(control_map)
        except Exception:
            pass

        self.param_widgets = new_param_widgets

        # No legacy fallbacks: the view exposes only the dynamic param_widgets mapping.
        # Consumers should read parameters from param_widgets or use the ViewModel API.

    # --------------------------
    # Config dialog (view-only)
    # --------------------------
    class _ConfigDialog(QDialog):
        def __init__(self, parent, cfg_dict):
            super().__init__(parent)
            self.setWindowTitle("Edit Configuration")
            form = QFormLayout(self)

            # default load folder
            self.load_edit = QLineEdit(self)
            self.load_edit.setText(str(cfg_dict.get("default_load_folder", "")))
            load_h = QHBoxLayout()
            load_h.addWidget(self.load_edit)
            browse_load = QPushButton("Browse")
            load_h.addWidget(browse_load)
            form.addRow("Default Load Folder:", load_h)
            browse_load.clicked.connect(self._browse_load)

            # default save folder
            self.save_edit = QLineEdit(self)
            self.save_edit.setText(str(cfg_dict.get("default_save_folder", "")))
            save_h = QHBoxLayout()
            save_h.addWidget(self.save_edit)
            browse_save = QPushButton("Browse")
            save_h.addWidget(browse_save)
            form.addRow("Default Save Folder:", save_h)
            browse_save.clicked.connect(self._browse_save)

            # buttons: Save, Cancel, and Reload (reload reads config from disk)
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            reload_btn = QPushButton("Reload")
            buttons.addButton(reload_btn, QDialogButtonBox.ActionRole)
            form.addRow(buttons)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            reload_btn.clicked.connect(self._on_reload_clicked)

        def _browse_load(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Load Folder", self.load_edit.text() or "")
            if d:
                self.load_edit.setText(d)

        def _browse_save(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Save Folder", self.save_edit.text() or "")
            if d:
                self.save_edit.setText(d)

        def _on_reload_clicked(self):
            """Reload configuration from disk via the parent ViewModel and refresh fields."""
            try:
                parent = self.parent()
                if parent and hasattr(parent, "viewmodel") and parent.viewmodel:
                    try:
                        parent.viewmodel.reload_config()
                    except Exception:
                        pass
                    try:
                        cfg = parent.viewmodel.get_config()
                        self.load_edit.setText(str(cfg.get("default_load_folder", "")))
                        self.save_edit.setText(str(cfg.get("default_save_folder", "")))
                        # also inform user in the main log
                        try:
                            parent.append_log("Configuration reloaded into dialog.")
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

    def _on_edit_config_clicked(self):
        """Open the configuration editor dialog (view-only)."""
        if not self.viewmodel:
            self.append_log("No ViewModel available to edit configuration.")
            return

        # Ask ViewModel for current config values (ViewModel handles logic)
        try:
            cfg = self.viewmodel.get_config()
        except Exception as e:
            self.append_log(f"Failed to load configuration: {e}")
            return

        dlg = self._ConfigDialog(self, cfg)
        if dlg.exec() == QDialog.Accepted:
            # collect values and instruct ViewModel to save them
            new_load = dlg.load_edit.text().strip()
            new_save = dlg.save_edit.text().strip()
            try:
                ok = self.viewmodel.save_config(default_load_folder=new_load, default_save_folder=new_save)
                if ok:
                    self.append_log("Configuration saved.")
                    # reload ViewModel config and refresh UI as needed
                    if hasattr(self.viewmodel, "reload_config"):
                        try:
                            self.viewmodel.reload_config()
                        except Exception:
                            pass
                else:
                    self.append_log("Configuration save failed.")
            except Exception as e:
                self.append_log(f"Error saving configuration: {e}")

    def _on_curve_clicked(self, event, curve_id):
        """Handle mouse click on a curve."""
        if not self.viewmodel:
            return
        # Set this as the selected curve
        self.viewmodel.set_selected_curve(curve_id)

    def _on_curve_selected(self, curve_id):
        """Highlight the selected curve visually."""
        # Deselect previous
        if self.selected_curve_id and self.selected_curve_id in self.curves:
            curve = self.curves[self.selected_curve_id]
            curve.setPen(pg.mkPen(FIT_COLOR, width=2))
        self.selected_curve_id = curve_id

        # Highlight new
        if curve_id and curve_id in self.curves:
            curve = self.curves[curve_id]
            curve.setPen(pg.mkPen('red', width=4))
        self.append_log(f"Curve selection changed → {curve_id or 'none'}")


