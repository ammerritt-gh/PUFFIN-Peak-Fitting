"""
Resolution dock widget for global resolution convolution settings.

This dock provides controls for selecting and configuring a resolution function
that will be convolved with the base model. The resolution is "global" in that
it applies to all model components.
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QScrollArea, QSpinBox, QCheckBox,
    QLineEdit, QHBoxLayout, QGroupBox
)
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg
import numpy as np
import math
from functools import partial


class ResolutionDock(QDockWidget):
    """Floating dock widget for global resolution convolution settings."""

    # Signals
    resolution_model_changed = Signal(str)  # model name or "None"
    resolution_parameter_changed = Signal(str, object)  # parameter name, value
    resolution_apply_clicked = Signal()

    def __init__(self, parent=None):
        """
        Initialize the resolution dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Global Resolution", parent)
        self.param_widgets = {}
        self._param_last_values = {}
        self._building_params = False
        self._control_map = {}
        
        # Allow floating and closing; dock retains state when closed
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.setFloating(True)
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # Model selector at top (with "None" option)
        self.model_selector = QComboBox()
        self.model_selector.addItem("None")  # No resolution convolution
        
        # Populate with available resolution models
        try:
            from models import get_available_model_names
            import re
            
            def _pretty(name: str) -> str:
                s = re.sub(r"ModelSpec$", "", name)
                s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
                pretty = s.strip()
                if pretty.lower() == "composite":
                    return "Custom Model"
                return pretty
            
            spec_class_names = get_available_model_names()
            display_names = [_pretty(n) for n in spec_class_names]
            # Filter out composite/custom for resolution (doesn't make sense)
            display_names = [n for n in display_names if n.lower() not in ("composite", "custom", "custom model")]
            if display_names:
                self.model_selector.addItems(display_names)
            else:
                # Fallback to static list
                self.model_selector.addItems(["Gaussian", "Voigt"])
        except Exception:
            # Fallback if discovery fails
            self.model_selector.addItems(["Gaussian", "Voigt"])
        
        vlayout.addWidget(QLabel("Resolution Model:"))
        vlayout.addWidget(self.model_selector)

        # Preview plot
        vlayout.addWidget(QLabel("Resolution Preview:"))
        self.preview_widget = pg.PlotWidget()
        self.preview_widget.setBackground("white")
        self.preview_widget.setMinimumHeight(150)
        self.preview_widget.setMaximumHeight(200)
        self.preview_widget.showGrid(x=True, y=True, alpha=0.3)
        # Create a plot item for the resolution curve
        self.preview_curve = self.preview_widget.plot(
            [], [], pen=pg.mkPen("blue", width=2)
        )
        vlayout.addWidget(self.preview_widget)

        # Scrollable parameters area
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_form_widget = QWidget()
        self.param_form = QFormLayout(self.param_form_widget)
        self.param_scroll.setWidget(self.param_form_widget)
        vlayout.addWidget(self.param_scroll)

        # Apply button
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        btn_layout.addWidget(self.apply_btn)
        vlayout.addLayout(btn_layout)

        self.setWidget(container)
        
        # Reasonable default size
        self.setMinimumWidth(320)
        self.resize(350, 500)

        # Connect signals
        self.model_selector.currentTextChanged.connect(self._on_model_changed)
        self.apply_btn.clicked.connect(self._on_apply_clicked)

    def _on_model_changed(self, model_name):
        """Handle resolution model selection change."""
        self.resolution_model_changed.emit(model_name)

    def _on_apply_clicked(self):
        """Handle apply button click."""
        self.resolution_apply_clicked.emit()

    def get_selected_model(self) -> str:
        """Get the currently selected resolution model name."""
        return self.model_selector.currentText()

    def set_model_selector(self, model_name: str):
        """Set the model selector to the specified model."""
        try:
            idx = self.model_selector.findText(str(model_name), Qt.MatchFixedString)
            if idx >= 0:
                self.model_selector.setCurrentIndex(idx)
        except Exception:
            pass

    def update_preview(self, x_data=None, y_data=None):
        """
        Update the resolution preview plot.

        Args:
            x_data: X values for the preview (optional, uses default if None)
            y_data: Y values for the resolution function
        """
        try:
            if x_data is None:
                x_data = np.linspace(-5, 5, 200)
            if y_data is None:
                y_data = np.zeros_like(x_data)
            
            self.preview_curve.setData(x_data, y_data)
        except Exception:
            pass

    def get_parameter_values(self) -> dict:
        """
        Collect values from parameter widgets.

        Returns:
            Dict mapping parameter names to values
        """
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
                    params[name] = getattr(widget, "value", lambda: None)()
            except Exception:
                params[name] = None
        return params

    def populate_parameters(self, specs: dict):
        """
        Populate the parameter form from specs dict.

        Args:
            specs: Dict of parameter specifications (same format as ParametersDock)
        """
        self._building_params = True
        try:
            # Create new widget for parameters
            new_widget = QWidget()
            outer_layout = QVBoxLayout(new_widget)
            outer_layout.setContentsMargins(8, 8, 8, 8)
            outer_layout.setSpacing(10)

            self.param_widgets = {}
            self._param_last_values = {}

            if not specs:
                # No parameters - show message
                label = QLabel("No parameters available.\nSelect a resolution model above.")
                label.setAlignment(Qt.AlignCenter)
                outer_layout.addWidget(label)
            else:
                group_box = QGroupBox("Resolution Parameters")
                form_layout = QFormLayout()
                form_layout.setContentsMargins(6, 12, 6, 6)
                form_layout.setSpacing(6)
                group_box.setLayout(form_layout)

                for name, meta in specs.items():
                    if isinstance(meta, dict):
                        spec_dict = meta
                    else:
                        spec_dict = {"value": meta}

                    val = spec_dict.get("value")
                    ptype = spec_dict.get("type")
                    
                    # Infer type if not specified
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
                            w.setRange(spec_dict.get("min", -1e9), spec_dict.get("max", 1e9))
                            w.setDecimals(spec_dict.get("decimals", 6))
                            w.setSingleStep(spec_dict.get("step", 0.1))
                            if val is not None:
                                w.blockSignals(True)
                                try:
                                    w.setValue(float(val))
                                finally:
                                    w.blockSignals(False)
                            widget = w
                        elif ptype == "int":
                            w = QSpinBox()
                            w.setRange(int(spec_dict.get("min", -2147483648)), int(spec_dict.get("max", 2147483647)))
                            w.setSingleStep(int(spec_dict.get("step", 1)))
                            if val is not None:
                                w.blockSignals(True)
                                try:
                                    w.setValue(int(val))
                                finally:
                                    w.blockSignals(False)
                            widget = w
                        elif ptype == "bool":
                            w = QCheckBox()
                            if val is not None:
                                w.blockSignals(True)
                                try:
                                    w.setChecked(bool(val))
                                finally:
                                    w.blockSignals(False)
                            widget = w
                        elif ptype == "choice":
                            w = QComboBox()
                            choices = spec_dict.get("choices", [])
                            if isinstance(val, (list, tuple)):
                                choices = list(val)
                            w.blockSignals(True)
                            try:
                                for c in choices:
                                    w.addItem(str(c))
                                cur = spec_dict.get("value")
                                if cur is not None:
                                    idx = w.findText(str(cur))
                                    if idx >= 0:
                                        w.setCurrentIndex(idx)
                            finally:
                                w.blockSignals(False)
                            widget = w
                        else:
                            w = QLineEdit()
                            if val is not None:
                                w.blockSignals(True)
                                try:
                                    w.setText(str(val))
                                finally:
                                    w.blockSignals(False)
                            widget = w
                    except Exception:
                        w = QLineEdit()
                        if val is not None:
                            w.blockSignals(True)
                            try:
                                w.setText(str(val))
                            finally:
                                w.blockSignals(False)
                        widget = w

                    # Set tooltip if hint provided
                    try:
                        hint = spec_dict.get("hint")
                        if hint:
                            widget.setToolTip(str(hint))
                    except Exception:
                        pass

                    # Size constraints for widgets
                    try:
                        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                            widget.setMaximumWidth(120)
                        elif isinstance(widget, QComboBox):
                            widget.setMaximumWidth(140)
                        elif isinstance(widget, QLineEdit):
                            widget.setMaximumWidth(140)
                    except Exception:
                        pass

                    # Create row with value widget + link + fixed controls
                    container_h = QWidget()
                    hbox = QHBoxLayout(container_h)
                    hbox.setContentsMargins(0, 0, 0, 0)
                    hbox.addWidget(widget)

                    # Link group spinbox
                    try:
                        link_val = spec_dict.get("link_group", None)
                        link_val = int(link_val) if link_val else 0
                    except Exception:
                        link_val = 0
                    link_spin = QSpinBox()
                    link_spin.setRange(0, 99)
                    link_spin.setValue(link_val)
                    link_spin.setToolTip("Link group (0 = not linked)")
                    link_spin.setFixedWidth(50)
                    self._bind_param_widget(f"{name}__link", link_spin)
                    hbox.addWidget(link_spin)

                    # Fixed checkbox
                    try:
                        fixed_val = bool(spec_dict.get("fixed", False))
                    except Exception:
                        fixed_val = False
                    fixed_chk = QCheckBox("Fix")
                    fixed_chk.setChecked(fixed_val)
                    try:
                        widget.setEnabled(not fixed_val)
                    except Exception:
                        pass
                    self._bind_param_widget(f"{name}__fixed", fixed_chk)
                    hbox.addWidget(fixed_chk)

                    hbox.addStretch(1)

                    form_layout.addRow(f"{name}:", container_h)
                    self._bind_param_widget(name, widget)

                outer_layout.addWidget(group_box)

            outer_layout.addStretch(1)

            # Replace scroll content
            self.param_scroll.takeWidget()
            self.param_scroll.setWidget(new_widget)
            self.param_form_widget = new_widget
            self.param_form = None

        finally:
            self._building_params = False

    def _bind_param_widget(self, name: str, widget):
        """Bind a parameter widget to track changes."""
        if widget is None:
            return
        self.param_widgets[name] = widget
        self._param_last_values[name] = self._read_param_widget(widget)

        if isinstance(widget, QDoubleSpinBox):
            try:
                widget.setKeyboardTracking(False)
            except Exception:
                pass
            widget.valueChanged.connect(partial(self._on_param_value_changed, name))
            widget.editingFinished.connect(partial(self._on_param_editing_finished, name))
        elif isinstance(widget, QSpinBox):
            try:
                widget.setKeyboardTracking(False)
            except Exception:
                pass
            widget.valueChanged.connect(partial(self._on_param_value_changed, name))
            widget.editingFinished.connect(partial(self._on_param_editing_finished, name))
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(partial(self._on_param_value_changed, name))
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(partial(self._on_param_value_changed, name))
        elif isinstance(widget, QLineEdit):
            widget.editingFinished.connect(partial(self._on_param_editing_finished, name))

    def _read_param_widget(self, widget):
        """Read the current value from a parameter widget."""
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return getattr(widget, "value", lambda: None)()

    def _on_param_value_changed(self, name, *args):
        """Handle parameter value change."""
        self._commit_parameter(name)

    def _on_param_editing_finished(self, name):
        """Handle parameter editing finished."""
        self._commit_parameter(name)

    def _commit_parameter(self, name: str):
        """Commit a parameter change by emitting a signal."""
        if self._building_params:
            return
        widget = self.param_widgets.get(name)
        if widget is None:
            return
        value = self._read_param_widget(widget)
        last = self._param_last_values.get(name, None)
        
        if isinstance(value, float) and isinstance(last, float):
            if math.isclose(value, last, rel_tol=1e-9, abs_tol=1e-9):
                return
        elif value == last:
            return

        # Handle fixed checkbox toggling the value widget
        if isinstance(name, str) and name.endswith("__fixed"):
            base = name[: -len("__fixed")]
            try:
                val_widget = self.param_widgets.get(base)
                if val_widget is not None:
                    val_widget.setEnabled(not bool(value))
            except Exception:
                pass

        self._param_last_values[name] = value
        self.resolution_parameter_changed.emit(name, value)

    def get_control_map(self):
        """Get the control map for interactive parameter controls."""
        return getattr(self, '_control_map', {})
