# view/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox,
    QDialog, QLineEdit, QDialogButtonBox, QHBoxLayout, QFileDialog,
    QScrollArea, QSpinBox, QCheckBox, QGroupBox, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QTimer
import pyqtgraph as pg
import numpy as np
from .input_handler import InputHandler

# -- color palette (change these) --
PLOT_BG = "white"       # plot background
POINT_COLOR = "black"   # scatter points
ERROR_COLOR = "black"   # error bars (can match points)
FIT_COLOR = "red"     # fit line
AXIS_COLOR = "black"    # axis and tick labels
GRID_ALPHA = 0.5

class MainWindow(QMainWindow):
    # Signal emitted when parameters are updated (for reconnecting signals)
    parameters_updated = Signal()
    
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")
        self.viewmodel = viewmodel
        self.param_widgets = {}   # name -> widget
        # debounce timers for auto-apply per-parameter
        self._debounce_timers = {}
        # debounce interval in milliseconds (tunable)
        # debounce interval in milliseconds (tunable)
        # 0 -> immediate apply; keep debounce available for future tuning
        self.auto_apply_debounce_ms = 0
        # whether auto-apply is enabled (can add UI toggle later)
        self.auto_apply_enabled = True
        # Last plotted arrays (x, y, y_fit) for selection logic
        self._last_plot_arrays = (None, None, None)
        # selected curve id/name (e.g., 'fit' or 'data')
        self._selected_curve = None
        # store original pen/visuals for curves so we can restore them
        self._curve_original_opts = {}

        # --- Central Plot ---
        self.plot_widget = pg.PlotWidget(title="Data and Fit")
        self.setCentralWidget(self.plot_widget)
        self._init_plot()
        
        # --- Input Handler ---
        self.input_handler = InputHandler(self.plot_widget)
        self._connect_input_handler()

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
        reload_cfg_btn = QPushButton("Reload Config")
        config_btn = QPushButton("Edit Config")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(fit_btn)
        layout.addWidget(reload_cfg_btn)
        layout.addWidget(config_btn)
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
            reload_cfg_btn.clicked.connect(getattr(self.viewmodel, "reload_config", lambda: None))
            config_btn.clicked.connect(self._on_edit_config_clicked)

    def _init_right_dock(self):
        # Replaced static parameter controls with a dynamic, scrollable form.
        self.right_dock = QDockWidget("Parameters", self)
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # Model selector placed at the top of the parameters panel
        self.model_selector = QComboBox()
        # Populate model selector dynamically from the models package when possible.
        try:
            from models import get_available_model_names
            names = get_available_model_names() or []
            # ensure a sensible fallback
            if not names:
                names = ["Voigt", "DHO+Voigt", "Gaussian", "DHO"]
        except Exception:
            names = ["Voigt", "DHO+Voigt", "Gaussian", "DHO"]
        self.model_selector.addItems(names)
        vlayout.addWidget(QLabel("Model:"))
        vlayout.addWidget(self.model_selector)
        
        # Add "New Model" button for creating custom models
        new_model_btn = QPushButton("New Model...")
        new_model_btn.setToolTip("Create a new custom composite model")
        new_model_btn.clicked.connect(self._on_new_model_clicked)
        vlayout.addWidget(new_model_btn)

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

        # Connect to ViewModel (if present)
        if self.viewmodel:
            apply_btn.clicked.connect(self._on_apply_clicked)
            refresh_btn.clicked.connect(self._refresh_parameters)
            # When the ViewModel reports parameter changes (e.g. after a fit), refresh UI
            try:
                if hasattr(self.viewmodel, "parameters_updated"):
                    self.viewmodel.parameters_updated.connect(self._refresh_parameters)
                    # Also refresh model selector when parameters update (custom models may be added)
                    self.viewmodel.parameters_updated.connect(self._refresh_model_selector)
            except Exception:
                pass
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
            yfit_arr = None
            self.fit_curve.clear()

        # store last plotted arrays for selection logic
        try:
            self._last_plot_arrays = (x_arr, y_arr, yfit_arr)
        except Exception:
            self._last_plot_arrays = (None, None, None)

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

    def _auto_apply_param(self, name, widget):
        """Auto-apply a single parameter from its widget when the widget value changes.

        This reads the new value from the widget and calls ViewModel.apply_parameters
        with a single-entry dict. Signals are connected only after widgets are
        initialized to avoid spurious triggers during panel population.
        """
        if not self.viewmodel:
            return
        try:
            # Extract value depending on widget type
            val = None
            if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                val = widget.value()
            elif isinstance(widget, QCheckBox):
                val = widget.isChecked()
            elif isinstance(widget, QComboBox):
                val = widget.currentText()
            elif isinstance(widget, QLineEdit):
                val = widget.text()
            else:
                # fallback: try a value() callable
                try:
                    val = getattr(widget, "value")()
                except Exception:
                    val = None

            # Apply single-parameter update via the ViewModel
            try:
                self.viewmodel.apply_parameters({name: val})
            except Exception as e:
                # do not raise; log instead
                try:
                    self.append_log(f"Auto-apply failed for {name}: {e}")
                except Exception:
                    pass
        except Exception:
            # be quiet on widget-read errors
            pass

    def _schedule_auto_apply(self, name, widget):
        """Schedule a debounced auto-apply for parameter `name`.

        Rapid widget changes will reset the timer; the ViewModel.apply will be
        invoked only after `self.auto_apply_debounce_ms` of inactivity.
        """
        if not getattr(self, "auto_apply_enabled", True):
            return
        try:
            # get existing timer or create one
            t = self._debounce_timers.get(name)
            if t is None:
                t = QTimer(self)
                t.setSingleShot(True)
                # connect to call the actual apply
                t.timeout.connect(lambda n=name, w=widget: self._auto_apply_param(n, w))
                self._debounce_timers[name] = t
            else:
                # update the connected widget reference by reconnecting on timeout
                try:
                    # disconnect previous and reconnect to use the latest widget
                    try:
                        t.timeout.disconnect()
                    except Exception:
                        pass
                    t.timeout.connect(lambda n=name, w=widget: self._auto_apply_param(n, w))
                except Exception:
                    pass

            # restart timer
            try:
                t.stop()
            except Exception:
                pass
            t.start(int(self.auto_apply_debounce_ms))
        except Exception:
            # fallback: immediate apply on any error
            try:
                self._auto_apply_param(name, widget)
            except Exception:
                pass

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
    
    def _refresh_model_selector(self):
        """Refresh the model selector dropdown with current models."""
        try:
            # Save current selection
            current_text = self.model_selector.currentText()
            
            # Clear and repopulate
            self.model_selector.clear()
            
            from models import get_available_model_names
            names = get_available_model_names() or []
            if not names:
                names = ["Voigt", "DHO+Voigt", "Gaussian", "DHO"]
            
            self.model_selector.addItems(names)
            
            # Restore selection if still available
            if current_text:
                idx = self.model_selector.findText(current_text, Qt.MatchFixedString)
                if idx >= 0:
                    self.model_selector.setCurrentIndex(idx)
        except Exception as e:
            self.append_log(f"Failed to refresh model selector: {e}")
    
    def _on_new_model_clicked(self):
        """Handle New Model button click."""
        if not self.viewmodel:
            return
        
        # Prompt for model name
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "New Custom Model", "Enter model name:")
        
        if ok and name:
            name = name.strip()
            if not name:
                self.append_log("Model name cannot be empty")
                return
            
            # Create the model
            success = self.viewmodel.create_custom_model(name)
            if success:
                # Refresh model selector and switch to new model
                self._refresh_model_selector()
                
                # Select the new model
                custom_name = f"Custom: {name}"
                idx = self.model_selector.findText(custom_name, Qt.MatchFixedString)
                if idx >= 0:
                    self.model_selector.setCurrentIndex(idx)
                
                self.append_log(f"Created custom model: {name}")
            else:
                self.append_log(f"Failed to create model (may already exist): {name}")

    def _populate_parameters(self, specs: dict):
        """Populate the parameter form from a specs dict.

        Each specs entry may be:
          name: value                      # infer type from value
          name: { 'value': v, 'type': 'float'|'int'|'str'|'bool'|'choice',
                  'min': ..., 'max': ..., 'choices': [...], 'decimals': ..., 'step': ..., 'input': ..., 'hint': ... }
        
        Special key 'groups': List of grouped parameters for composite models
          [{"label": "Group 1", "params": {...}, "component_index": 0, "base_spec": "gaussian"}, ...]
        
        Note: 'decimals' controls the number of decimal places shown by the
        QDoubleSpinBox (display precision). 'step' controls the single-step
        increment used when the user clicks the up/down arrows.
        """
        # Build a fresh form widget and replace the scroll area's widget
        new_widget = QWidget()
        new_form = QVBoxLayout(new_widget)  # Changed to VBoxLayout for groupboxes
        new_param_widgets = {}
        
        # Check if this is a grouped parameter spec (for composite models)
        if "groups" in specs and isinstance(specs["groups"], list):
            groups = specs["groups"]
            
            # Add buttons for custom model management if applicable
            if self.viewmodel and self.viewmodel.is_custom_model_active():
                control_widget = self._create_custom_model_controls()
                if control_widget:
                    new_form.addWidget(control_widget)
            
            # Create a group box for each component
            for group_info in groups:
                label = group_info.get("label", "Component")
                group_params = group_info.get("params", {})
                component_index = group_info.get("component_index", -1)
                base_spec = group_info.get("base_spec", "")
                
                # Create group box
                from PySide6.QtWidgets import QGroupBox
                group_box = QGroupBox(label)
                group_layout = QFormLayout(group_box)
                
                # Add remove/move buttons for this component
                if self.viewmodel and self.viewmodel.is_custom_model_active() and component_index >= 0:
                    btn_layout = QHBoxLayout()
                    
                    move_up_btn = QPushButton("↑")
                    move_up_btn.setMaximumWidth(30)
                    move_up_btn.setToolTip("Move component up")
                    
                    move_down_btn = QPushButton("↓")
                    move_down_btn.setMaximumWidth(30)
                    move_down_btn.setToolTip("Move component down")
                    
                    remove_btn = QPushButton("Remove")
                    remove_btn.setToolTip("Remove this component")
                    
                    # Connect buttons
                    model_name = self._get_active_custom_model_name()
                    if model_name:
                        move_up_btn.clicked.connect(
                            lambda checked, idx=component_index, name=model_name: 
                            self._move_component_up(name, idx))
                        move_down_btn.clicked.connect(
                            lambda checked, idx=component_index, name=model_name: 
                            self._move_component_down(name, idx))
                        remove_btn.clicked.connect(
                            lambda checked, idx=component_index, name=model_name: 
                            self._remove_component(name, idx))
                    
                    btn_layout.addWidget(move_up_btn)
                    btn_layout.addWidget(move_down_btn)
                    btn_layout.addWidget(remove_btn)
                    btn_layout.addStretch()
                    
                    group_layout.addRow(btn_layout)
                
                # Add parameters for this group
                for param_name, param_spec in group_params.items():
                    # Prefix the parameter name with the group label for widget storage
                    prefixed_name = f"{label}::{param_name}"
                    widget = self._create_parameter_widget(param_name, param_spec, prefixed_name)
                    if widget:
                        group_layout.addRow(param_name + ":", widget)
                        new_param_widgets[prefixed_name] = widget
                
                new_form.addWidget(group_box)
        
        else:
            # Standard non-grouped parameters - use form layout
            form_widget = QWidget()
            form_layout = QFormLayout(form_widget)
            
            for name, spec in specs.items():
                # Skip the special "groups" key if present
                if name == "groups":
                    continue
                
                widget = self._create_parameter_widget(name, spec, name)
                if widget:
                    # Attach hints/tooltip and input hint display
                    input_hint = None
                    if isinstance(spec, dict):
                        input_hint = spec.get("input") or spec.get("input_hint")
                        help_hint = spec.get("hint")
                    else:
                        help_hint = None

                    # set tooltip from help hint and input hint (string or structured)
                    try:
                        tt = help_hint or ""
                        display_hint = ""
                        if input_hint:
                            # if structured dict, produce a concise human-readable summary
                            if isinstance(input_hint, dict):
                                parts = []
                                for k, v in input_hint.items():
                                    if k == "wheel":
                                        mods = ",".join(v.get("modifiers", [])) if isinstance(v, dict) else ""
                                        parts.append(f"Wheel{(' + ' + mods) if mods else ''}: {v.get('action')}")
                                    elif k == "drag":
                                        parts.append(f"Drag: {v.get('action')}")
                                    elif k == "hotkey":
                                        parts.append(f"Hotkey: {v.get('key') if isinstance(v, dict) else str(v)}")
                                    else:
                                        parts.append(f"{k}: {str(v)}")
                                display_hint = "; ".join(parts)
                            else:
                                display_hint = str(input_hint)
                            if display_hint:
                                tt = (tt + "\n" + display_hint).strip()
                        if tt and widget is not None:
                            widget.setToolTip(tt)
                    except Exception:
                        pass

                    # If there's an input hint, show a small label beside the control
                    if input_hint:
                        container = QWidget()
                        h = QHBoxLayout(container)
                        h.setContentsMargins(0, 0, 0, 0)
                        h.addWidget(widget)
                        # show the display_hint (from above) or a short textual fallback
                        hint_text = display_hint if 'display_hint' in locals() and display_hint else (str(input_hint) if not isinstance(input_hint, dict) else "interactive")
                        hint_label = QLabel(hint_text)
                        hint_label.setToolTip(widget.toolTip() or hint_text)
                        try:
                            hint_label.setStyleSheet("color: gray; font-size: 11px;")
                        except Exception:
                            pass
                        h.addWidget(hint_label)
                        form_layout.addRow(name + ":", container)
                    else:
                        form_layout.addRow(name + ":", widget)

                    new_param_widgets[name] = widget

            new_form.addWidget(form_widget)
            new_form.addStretch()

        # Replace the widget shown in the scroll area (frees old widgets)
        self.param_scroll.takeWidget()
        self.param_scroll.setWidget(new_widget)
        self.param_form_widget = new_widget
        self.param_widgets = new_param_widgets
    
    def _create_parameter_widget(self, name: str, spec, storage_name: str):
        """Create a widget for a parameter. Returns the widget or None on error.
        
        Args:
            name: Display name of the parameter
            spec: Parameter specification dict or value
            storage_name: Name to use for storing the widget (may be prefixed for groups)
        """
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
                w.setRange(spec.get("min", -1e9) if isinstance(spec, dict) else -1e9, 
                          spec.get("max", 1e9) if isinstance(spec, dict) else 1e9)
                # 'decimals' controls how many decimal places the spinbox displays.
                w.setDecimals(spec.get("decimals", 6) if isinstance(spec, dict) else 6)
                # 'step' controls the increment when the user clicks arrows or presses keys
                w.setSingleStep(spec.get("step", 0.1) if isinstance(spec, dict) else 0.1)
                if val is not None:
                    w.setValue(float(val))
                widget = w

            elif ptype == "int":
                w = QSpinBox()
                w.setRange(int(spec.get("min", -2147483648) if isinstance(spec, dict) else -2147483648), 
                          int(spec.get("max", 2147483647) if isinstance(spec, dict) else 2147483647))
                w.setSingleStep(int(spec.get("step", 1) if isinstance(spec, dict) else 1))
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

            # Connect widget change signals to auto-apply (connect after initial value set)
            if widget:
                try:
                    # connect to a scheduler that debounces rapid changes
                    if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                        widget.valueChanged.connect(lambda v, n=storage_name, w=widget: self._schedule_auto_apply(n, w))
                    elif isinstance(widget, QCheckBox):
                        widget.stateChanged.connect(lambda s, n=storage_name, w=widget: self._schedule_auto_apply(n, w))
                    elif isinstance(widget, QComboBox):
                        widget.currentIndexChanged.connect(lambda i, n=storage_name, w=widget: self._schedule_auto_apply(n, w))
                    elif isinstance(widget, QLineEdit):
                        widget.textChanged.connect(lambda t, n=storage_name, w=widget: self._schedule_auto_apply(n, w))
                except Exception:
                    pass

        except Exception:
            # on any error, fallback to a simple line edit
            w = QLineEdit()
            if val is not None:
                w.setText(str(val))
            # set tooltip if available
            try:
                if isinstance(spec, dict):
                    t = spec.get("hint") or spec.get("input") or ""
                    if t:
                        w.setToolTip(t)
            except Exception:
                pass
            widget = w
            # Connect fallback widget signals
            try:
                widget.textChanged.connect(lambda t, n=storage_name, w=widget: self._schedule_auto_apply(n, w))
            except Exception:
                pass

        return widget
    
    # --------------------------
    # Custom Model Management Helpers
    # --------------------------
    
    def _create_custom_model_controls(self):
        """Create control buttons for custom model management."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        add_btn = QPushButton("Add Element")
        add_btn.clicked.connect(self._on_add_element_clicked)
        layout.addWidget(add_btn)
        
        layout.addStretch()
        
        return widget
    
    def _get_active_custom_model_name(self):
        """Extract the custom model name from the active model."""
        if not self.viewmodel:
            return None
        
        model_name = getattr(self.viewmodel.state, "model_name", "")
        name_lower = model_name.lower().strip()
        
        if name_lower.startswith("custom:") or name_lower.startswith("custom "):
            if ":" in model_name:
                return model_name.split(":", 1)[1].strip()
            elif " " in model_name:
                return model_name.split(None, 1)[1].strip()
        
        return None
    
    def _on_add_element_clicked(self):
        """Handle Add Element button click."""
        if not self.viewmodel:
            return
        
        model_name = self._get_active_custom_model_name()
        if not model_name:
            self.append_log("No custom model is currently active")
            return
        
        # Get list of addable models
        try:
            from models import get_available_model_names
            addable_models = get_available_model_names(addable_only=True)
        except Exception as e:
            self.append_log(f"Failed to get model list: {e}")
            return
        
        if not addable_models:
            self.append_log("No addable models available")
            return
        
        # Show dialog to select model
        from PySide6.QtWidgets import QInputDialog
        model_type, ok = QInputDialog.getItem(
            self, "Add Element", "Select model type to add:", 
            addable_models, 0, False)
        
        if ok and model_type:
            # Get the canonical key for the selected model
            try:
                from models import MODEL_REGISTRY
                base_spec = None
                for entry in MODEL_REGISTRY:
                    if entry.get("display") == model_type or entry.get("key") == model_type:
                        base_spec = entry["key"]
                        break
                
                if base_spec:
                    success = self.viewmodel.add_component_to_custom_model(model_name, base_spec)
                    if success:
                        self.append_log(f"Added {model_type} to {model_name}")
                else:
                    self.append_log(f"Could not find model spec for {model_type}")
            except Exception as e:
                self.append_log(f"Error adding element: {e}")
    
    def _remove_component(self, model_name: str, index: int):
        """Remove a component from the custom model."""
        if not self.viewmodel:
            return
        
        # Confirm removal
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Remove component {index + 1}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            success = self.viewmodel.remove_component_from_custom_model(model_name, index)
            if success:
                self.append_log(f"Removed component {index + 1}")
    
    def _move_component_up(self, model_name: str, index: int):
        """Move a component up in the list."""
        if not self.viewmodel or index <= 0:
            return
        
        success = self.viewmodel.move_component_in_custom_model(model_name, index, index - 1)
        if success:
            self.append_log(f"Moved component {index + 1} up")
    
    def _move_component_down(self, model_name: str, index: int):
        """Move a component down in the list."""
        if not self.viewmodel:
            return
        
        # Get the number of components
        try:
            components = self.viewmodel.get_custom_model_components(model_name)
            if index >= len(components) - 1:
                return
        except Exception:
            return
        
        success = self.viewmodel.move_component_in_custom_model(model_name, index, index + 1)
        if success:
            self.append_log(f"Moved component {index + 1} down")

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

            # buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            form.addRow(buttons)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)

        def _browse_load(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Load Folder", self.load_edit.text() or "")
            if d:
                self.load_edit.setText(d)

        def _browse_save(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Save Folder", self.save_edit.text() or "")
            if d:
                self.save_edit.setText(d)

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
    
    # --------------------------
    # Input Handler Integration
    # --------------------------
    def _connect_input_handler(self):
        """Connect input_handler signals to view methods."""
        if not hasattr(self, 'input_handler') or self.input_handler is None:
            return
        
        try:
            # Connect mouse events
            self.input_handler.mouse_clicked.connect(self._on_plot_clicked)
            self.input_handler.mouse_moved.connect(self._on_plot_mouse_moved)
            # Debug: connect press/release/drag signals for logging
            try:
                self.input_handler.mouse_pressed.connect(self._on_mouse_pressed)
                self.input_handler.mouse_released.connect(self._on_mouse_released)
                self.input_handler.mouse_dragged.connect(self._on_mouse_dragged)
            except Exception:
                pass
            
            # Connect keyboard events
            self.input_handler.key_pressed.connect(self._on_plot_key_pressed)
            
            # Connect wheel events
            self.input_handler.wheel_scrolled.connect(self._on_plot_wheel_scrolled)
            
            self.append_log("Input handler connected successfully.")
        except Exception as e:
            self.append_log(f"Failed to connect input handler: {e}")

    # --------------------------
    # Debug mouse handlers
    # --------------------------
    def _btn_name(self, b):
        try:
            if b == Qt.LeftButton:
                return "Left"
            if b == Qt.RightButton:
                return "Right"
            if b == Qt.MidButton or b == Qt.MiddleButton:
                return "Middle"
        except Exception:
            pass
        return str(b)

    def _on_mouse_pressed(self, x, y, button, items):
        try:
            items_text = ", ".join(items) if items else "None"
            bname = self._btn_name(button)
            self.append_log(f"Mouse pressed: btn={bname}, pos=({x:.3f}, {y:.3f}), hit={items_text}")
        except Exception:
            pass

    def _on_mouse_released(self, x, y, button, items):
        try:
            items_text = ", ".join(items) if items else "None"
            bname = self._btn_name(button)
            self.append_log(f"Mouse released: btn={bname}, pos=({x:.3f}, {y:.3f}), hit={items_text}")
        except Exception:
            pass

    def _on_mouse_dragged(self, dx, dy, dist):
        try:
            self.append_log(f"Mouse dragged: dx={dx:.4g}, dy={dy:.4g}, dist={dist:.4g}")
        except Exception:
            pass
    
    def _on_plot_clicked(self, x, y, button):
        """
        Handle plot click events.
        
        Args:
            x: X coordinate in data space
            y: Y coordinate in data space  
            button: Mouse button used
        """
        try:
            # Log the click
            self.append_log(f"Plot clicked at ({x:.2f}, {y:.2f})")

            # First, try to select a plotted curve near the click (fit/data)
            try:
                picked_curve = False
                try:
                    picked_curve = self._try_select_curve(x, y)
                except Exception:
                    picked_curve = False
                if picked_curve:
                    return
            except Exception:
                pass

            # If no curve selected, try to select a parameter near the click if viewmodel is present.
            # If selection succeeds we enter selection mode; otherwise clear any selection.
            try:
                picked = False
                if self.viewmodel:
                    picked = self._try_select_param(x, y, button)
                if picked:
                    # if a curve was previously selected, clear its highlight
                    try:
                        if getattr(self, '_selected_curve', None) is not None:
                            try:
                                self._highlight_curve(self._selected_curve, False)
                            except Exception:
                                pass
                            self._selected_curve = None
                    except Exception:
                        pass
                else:
                    # clicking away: only clear selection if no curve is currently selected.
                    # (Requirement: clicking empty space should NOT deselect a curve.)
                    if getattr(self, '_selected_curve', None) is None:
                        try:
                            self._clear_selection()
                        except Exception:
                            pass
            except Exception:
                pass

            # Notify viewmodel of the click as well (viewmodel may need it)
            if self.viewmodel and hasattr(self.viewmodel, 'handle_plot_click'):
                self.viewmodel.handle_plot_click(x, y, button)
        except Exception as e:
            self.append_log(f"Error handling plot click: {e}")

    def _try_select_param(self, x, y, button):
        """Attempt to select a parameter for interactive input.

        Selection criteria:
          - parameter spec must include a 'drag' input action
          - parameter must have a numeric 'value' to compare against click x
          - click must be within a threshold of the parameter value (threshold ~ 2% of x-range)

        Returns True if selection started; False otherwise.
        """
        if not self.viewmodel:
            return False
        try:
            specs = getattr(self.viewmodel, "get_parameters", lambda: {})() or {}
            # compute x-range from data if available
            xdata = getattr(self.viewmodel.state, "x_data", None)
            try:
                if xdata is not None and len(xdata) >= 2:
                    xr = float(np.max(xdata) - np.min(xdata))
                else:
                    xr = 1.0
            except Exception:
                xr = 1.0
            # threshold: 2% of x-range or a small absolute lower bound
            threshold = max(0.02 * xr, 1e-3)

            closest = None
            best_dist = None
            for pname, pspec in specs.items():
                try:
                    if not isinstance(pspec, dict):
                        continue
                    inp = pspec.get("input") or pspec.get("input_hint")
                    if not inp or not isinstance(inp, dict):
                        continue
                    # only consider parameters that declare 'drag'
                    if "drag" not in inp:
                        continue
                    val = pspec.get("value")
                    if val is None:
                        continue
                    # must be numeric
                    try:
                        vnum = float(val)
                    except Exception:
                        continue
                    dist = abs(float(x) - vnum)
                    if dist <= threshold and (best_dist is None or dist < best_dist):
                        best_dist = dist
                        closest = pname
                except Exception:
                    continue

            if closest is None:
                return False

            # Ask ViewModel to begin selection for this parameter
            try:
                started = False
                if hasattr(self.viewmodel, "begin_selection"):
                    started = self.viewmodel.begin_selection(closest, x, y)
                if not started:
                    # As fallback, set a simple interactive drag flag in viewmodel state
                    try:
                        self.viewmodel.state._selected_param = closest
                        self.viewmodel.state._interactive_drag_info = {"handlers": [], "last_x": float(x), "last_y": float(y)}
                        started = True
                    except Exception:
                        started = False
                if started:
                    self._set_selection_active(closest)
                    return True
            except Exception:
                pass
        except Exception:
            pass
        return False

    def _try_select_curve(self, x, y):
        """Attempt to select a plotted curve (fit line or data scatter) near (x,y).
        Returns True if a curve was selected.
        """
        try:
            x_arr, y_arr, yfit_arr = getattr(self, "_last_plot_arrays", (None, None, None))
            # prefer fit curve if present
            if yfit_arr is not None and x_arr is not None and len(x_arr) >= 2:
                try:
                    # nearest index in x
                    idx = int(np.argmin(np.abs(x_arr - float(x))))
                    fit_y = float(yfit_arr[idx])
                    # compute a threshold from y-range
                    try:
                        yr = float(np.nanmax(y_arr) - np.nanmin(y_arr))
                    except Exception:
                        yr = max(abs(fit_y), 1.0)
                    threshold = max(0.02 * yr, 1e-6)
                    if abs(float(y) - fit_y) <= threshold:
                        # select fit curve
                        self._selected_curve = 'fit'
                        # highlight selection
                        try:
                            self._highlight_curve('fit', True)
                        except Exception:
                            pass
                        self._set_selection_active('fit')
                        return True
                except Exception:
                    pass

            # fallback: check scatter nearest point (Euclidean)
            if y_arr is not None and x_arr is not None and len(x_arr) >= 1:
                try:
                    idx = int(np.argmin(np.abs(x_arr - float(x))))
                    dx = float(x) - float(x_arr[idx])
                    dy = float(y) - float(y_arr[idx])
                    # use combined scale based on data ranges
                    try:
                        xr = float(np.nanmax(x_arr) - np.nanmin(x_arr))
                        yr = float(np.nanmax(y_arr) - np.nanmin(y_arr))
                    except Exception:
                        xr = yr = 1.0
                    # scaled distance
                    dist = np.hypot(dx / max(xr, 1e-9), dy / max(yr, 1e-9))
                    if dist <= 0.02:
                        self._selected_curve = 'data'
                        try:
                            self._highlight_curve('data', True)
                        except Exception:
                            pass
                        self._set_selection_active('data')
                        return True
                except Exception:
                    pass

        except Exception:
            pass
        return False

    def _highlight_curve(self, name, enable=True):
        """Visually highlight or un-highlight a named curve ('fit' or 'data')."""
        try:
            if name == 'fit':
                item = getattr(self, 'fit_curve', None)
                if item is None:
                    return
                # store original pen
                try:
                    if 'fit' not in self._curve_original_opts:
                        self._curve_original_opts['fit'] = {'pen': item.opts.get('pen') if hasattr(item, 'opts') else None}
                except Exception:
                    pass
                if enable:
                    try:
                        item.setPen(pg.mkPen('blue', width=4))
                    except Exception:
                        pass
                else:
                    # restore
                    try:
                        orig = self._curve_original_opts.get('fit', {}).get('pen')
                        if orig is not None:
                            item.setPen(orig)
                        else:
                            item.setPen(pg.mkPen(FIT_COLOR, width=2))
                    except Exception:
                        pass

            elif name == 'data':
                item = getattr(self, 'scatter', None)
                if item is None:
                    return
                try:
                    if 'data' not in self._curve_original_opts:
                        self._curve_original_opts['data'] = {'size': getattr(item, 'size', None)}
                except Exception:
                    pass
                if enable:
                    try:
                        # increase point size to indicate selection
                        item.setSize(10)
                    except Exception:
                        pass
                else:
                    try:
                        orig = self._curve_original_opts.get('data', {}).get('size')
                        if orig is not None:
                            item.setSize(orig)
                        else:
                            item.setSize(6)
                    except Exception:
                        pass
        except Exception:
            pass

    def _set_selection_active(self, pname):
        """Mark UI as in selection mode for a parameter and disable ViewBox mouse interactions."""
        try:
            self.append_log(f"Selected: {pname}")
            try:
                vb = self.plot_widget.getViewBox()
                if vb is not None:
                    # disable mouse panning/zooming so drag/wheel go to parameter control
                    vb.setMouseEnabled(False, False)
                    # optionally change cursor (kept simple)
            except Exception:
                pass
        except Exception:
            pass

    def _clear_selection(self):
        """Clear interactive selection and restore normal UI (panning/zoom enabled)."""
        try:
            # instruct viewmodel to end selection if it has such API
            try:
                if self.viewmodel and hasattr(self.viewmodel, "end_selection"):
                    self.viewmodel.end_selection()
                else:
                    # clear any state flags if fallback used
                    if self.viewmodel:
                        try:
                            if hasattr(self.viewmodel.state, "_selected_param"):
                                delattr(self.viewmodel.state, "_selected_param")
                        except Exception:
                            try:
                                setattr(self.viewmodel.state, "_selected_param", None)
                            except Exception:
                                pass
                            pass
                        try:
                            if hasattr(self.viewmodel.state, "_interactive_drag_info"):
                                delattr(self.viewmodel.state, "_interactive_drag_info")
                        except Exception:
                            try:
                                setattr(self.viewmodel.state, "_interactive_drag_info", None)
                            except Exception:
                                pass
            except Exception:
                pass

            # re-enable viewbox interactions
            try:
                vb = self.plot_widget.getViewBox()
                if vb is not None:
                    vb.setMouseEnabled(True, True)
            except Exception:
                pass

            # if a curve was selected, remove highlighting
            try:
                if getattr(self, '_selected_curve', None) is not None:
                    try:
                        self._highlight_curve(self._selected_curve, False)
                    except Exception:
                        pass
                    self._selected_curve = None
            except Exception:
                pass

            self.append_log("Selection cleared. UI interactions restored.")
        except Exception:
            pass

    def _on_plot_key_pressed(self, key, modifiers):
        """
        Handle keyboard events on the plot.

        Reserve Space to clear selection and restore UI.
        """
        try:
            # Space = deselect / revert to normal UI
            if key == Qt.Key_Space:
                self._clear_selection()
                # also forward a simple message to viewmodel if desired
                try:
                    if self.viewmodel and hasattr(self.viewmodel, "handle_key_press"):
                        self.viewmodel.handle_key_press(key, modifiers)
                except Exception:
                    pass
                return

            # other keys: same as before
            if key == Qt.Key_R:
                self.append_log("Reset view (R key)")
                try:
                    self.plot_widget.getViewBox().autoRange()
                except Exception:
                    pass
            else:
                if self.viewmodel and hasattr(self.viewmodel, 'handle_key_press'):
                    self.viewmodel.handle_key_press(key, modifiers)
        except Exception as e:
            self.append_log(f"Error handling key press: {e}")
    
    def _on_plot_mouse_moved(self, x, y, buttons=None):
        """
        Handle plot mouse move events.

        Args:
            x: X coordinate in data space
            y: Y coordinate in data space
            buttons: mouse button state (optional)
        """
        # Can be used for live cursor position display or dragging operations
        # For now, pass to viewmodel if it has a handler (include buttons)
        try:
            if self.viewmodel and hasattr(self.viewmodel, 'handle_plot_mouse_move'):
                # Some existing code paths may expect two args — call defensively
                try:
                    self.viewmodel.handle_plot_mouse_move(x, y, buttons)
                except TypeError:
                    # fallback to prior signature
                    self.viewmodel.handle_plot_mouse_move(x, y)
        except Exception:
            pass
    
    def _on_plot_wheel_scrolled(self, delta, modifiers):
        """
        Handle mouse wheel events on the plot.
        
        Args:
            delta: Wheel delta (positive = up, negative = down)
            modifiers: Qt keyboard modifiers
        """
        try:
            # Determine if modifiers are pressed
            is_ctrl = bool(modifiers & Qt.ControlModifier)
            is_shift = bool(modifiers & Qt.ShiftModifier)
            is_alt = bool(modifiers & Qt.AltModifier)
            
            # Log for debugging
            mods_str = []
            if is_ctrl:
                mods_str.append("Ctrl")
            if is_shift:
                mods_str.append("Shift")
            if is_alt:
                mods_str.append("Alt")
            mods_text = "+".join(mods_str) if mods_str else "None"
            
            self.append_log(f"Wheel scrolled: delta={delta}, modifiers={mods_text}")
            
            # Pass to viewmodel for parameter adjustments
            if self.viewmodel and hasattr(self.viewmodel, 'handle_wheel_scroll'):
                self.viewmodel.handle_wheel_scroll(delta, modifiers)
        except Exception as e:
            self.append_log(f"Error handling wheel scroll: {e}")
