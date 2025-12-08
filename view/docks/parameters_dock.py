"""
Parameters dock widget for model parameter controls.
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QScrollArea, QSpinBox, QCheckBox,
    QLineEdit, QHBoxLayout, QGroupBox
)
from PySide6.QtCore import Qt, Signal
import math
import re
from functools import partial


# Fit curve color (used for default styling)
FIT_COLOR = "purple"


class ParametersDock(QDockWidget):
    """Dock widget for managing model parameters."""

    # Signals
    model_changed = Signal(str)  # model name
    apply_clicked = Signal()
    refresh_clicked = Signal()
    load_custom_model_clicked = Signal()  # load saved custom model
    parameter_changed = Signal(str, object)  # parameter name, value
    parameters_updated = Signal()  # emitted when parameter panel is rebuilt

    def __init__(self, parent=None):
        """
        Initialize the parameters dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Parameters", parent)
        self.param_widgets = {}   # name -> widget
        self._param_last_values = {}
        self._building_params = False
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # Model selector placed at the top of the parameters panel
        self.model_selector = QComboBox()
        # Populate model selector dynamically from discovered model specs where possible.
        try:
            # lazy import to avoid circular imports at module import time
            from models import get_available_model_names

            def _pretty(name: str) -> str:
                # remove trailing 'ModelSpec' and split CamelCase into words
                s = re.sub(r"ModelSpec$", "", name)
                s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
                pretty = s.strip()
                if pretty.lower() == "composite":
                    return "Custom Model"
                return pretty

            spec_class_names = get_available_model_names()
            display_names = [_pretty(n) for n in spec_class_names]
            # ensure we have at least the historical defaults as fallback
            if display_names:
                self.model_selector.addItems(display_names)
            else:
                self.model_selector.addItems(["Voigt", "Gaussian"])
        except Exception:
            # Fall back to the original static list if discovery fails
            self.model_selector.addItems(["Voigt", "Gaussian"])
        vlayout.addWidget(QLabel("Model:"))
        vlayout.addWidget(self.model_selector)

        # Load Custom Model button
        self.load_custom_model_btn = QPushButton("Load Custom Model...")
        vlayout.addWidget(self.load_custom_model_btn)

        # Chi-squared display (placed above the parameter list)
        self.chi_label = QLabel("Chi-squared: N/A")
        try:
            f = self.chi_label.font()
            f.setPointSize(max(8, f.pointSize() - 1))
            self.chi_label.setFont(f)
        except Exception:
            pass
        vlayout.addWidget(self.chi_label)

        # Scrollable area to hold the form (so many parameters fit)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)

        # initial empty form widget (will be replaced by populate_parameters)
        self.param_form_widget = QWidget()
        self.param_form = QFormLayout(self.param_form_widget)
        self.param_scroll.setWidget(self.param_form_widget)

        vlayout.addWidget(self.param_scroll)

        # Apply + Refresh buttons
        btn_h = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.refresh_btn = QPushButton("Refresh")
        btn_h.addWidget(self.apply_btn)
        btn_h.addWidget(self.refresh_btn)
        vlayout.addLayout(btn_h)

        self.setWidget(container)

        # Make the parameters dock wider by default so controls and hints are visible
        try:
            # A minimum width allows the user to resize smaller/larger while
            # providing a comfortable default layout on startup.
            self.setMinimumWidth(360)
        except Exception:
            pass

        # Connect internal signals
        self.model_selector.currentIndexChanged.connect(self._on_model_selected)
        self.load_custom_model_btn.clicked.connect(self._on_load_custom_model_clicked)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)

    def _on_model_selected(self, idx):
        """Handle model selection change."""
        name = self.model_selector.currentText()
        self.model_changed.emit(name)

    def _on_load_custom_model_clicked(self):
        """Handle load custom model button click."""
        self.load_custom_model_clicked.emit()

    def _on_apply_clicked(self):
        """Handle apply button click."""
        self.apply_clicked.emit()

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        self.refresh_clicked.emit()

    def set_model_selector(self, model_name):
        """
        Set the model selector to the specified model.

        Args:
            model_name: Name of the model to select
        """
        try:
            idx = self.model_selector.findText(str(model_name), Qt.MatchFixedString)
            if idx >= 0:
                self.model_selector.setCurrentIndex(idx)
        except Exception:
            pass

    def update_chi_label(self, text):
        """
        Update the chi-squared label.

        Args:
            text: Text to display
        """
        try:
            self.chi_label.setText(text)
        except Exception:
            pass

    def get_parameter_values(self):
        """
        Collect values dynamically from widgets into a dict.

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
                    # fallback: try to read "value" attribute if a custom widget
                    params[name] = getattr(widget, "value", lambda: None)()
            except Exception:
                params[name] = None
        return params

    def populate_parameters(self, specs: dict):
        """
        Populate the parameter form from a specs dict.

        Each specs entry may be:
          name: value                      # infer type from value
          name: { 'value': v, 'type': 'float'|'int'|'str'|'bool'|'choice',
                  'min': ..., 'max': ..., 'choices': [...], 'decimals': ..., 'step': ... }
        Note: 'decimals' controls the number of decimal places shown by the
        QDoubleSpinBox (display precision). 'step' controls the single-step
        increment used when the user clicks the up/down arrows.

        Args:
            specs: Dict of parameter specifications
        """
        self._building_params = True
        try:
            normalized_specs = {}
            grouped: dict = {}
            group_order = []
            for name, spec in specs.items():
                if isinstance(spec, dict):
                    meta = dict(spec)
                else:
                    meta = {"value": spec}
                normalized_specs[name] = meta
                component_key = meta.get("component") if isinstance(meta, dict) else None
                if component_key not in grouped:
                    grouped[component_key] = []
                    group_order.append(component_key)
                grouped[component_key].append((name, meta))

            if None in grouped:
                group_order = [key for key in group_order if key is None] + [key for key in group_order if key is not None]

            new_widget = QWidget()
            outer_layout = QVBoxLayout(new_widget)
            outer_layout.setContentsMargins(8, 8, 8, 8)
            outer_layout.setSpacing(10)

            self.param_widgets = {}
            self._param_last_values = {}

            for group_key in group_order:
                entries = grouped.get(group_key, [])
                if not entries:
                    continue
                sample_meta = entries[0][1] if entries else {}
                if group_key is None:
                    label = "Model Parameters" if len(group_order) > 1 else "Parameters"
                    color = "#666666"
                else:
                    label = sample_meta.get("component_label") or str(group_key).rstrip("_") or str(group_key)
                    color = sample_meta.get("color") or FIT_COLOR
                group_box = QGroupBox(label)
                style = (
                    f"QGroupBox {{ border: 2px solid {color}; border-radius: 6px; margin-top: 8px; padding: 8px; }}"
                    f" QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: {color}; font-weight: bold; }}"
                )
                try:
                    group_box.setStyleSheet(style)
                except Exception as e:
                    print(f"Failed to set style for group box '{label}': {e}")

                form_layout = QFormLayout()
                form_layout.setContentsMargins(6, 12, 6, 6)
                form_layout.setSpacing(6)
                group_box.setLayout(form_layout)

                # Header row for the input area: helps indicate Link/Fix/Hint columns
                try:
                    header_widget = QWidget()
                    header_h = QHBoxLayout(header_widget)
                    header_h.setContentsMargins(0, 0, 0, 0)
                    header_h.addWidget(QLabel("Value"))
                    header_h.addStretch(1)
                    # Right-side header container matches the fixed width used for rows
                    right_header = QWidget()
                    rh_layout = QHBoxLayout(right_header)
                    rh_layout.setContentsMargins(0, 0, 0, 0)
                    rh_layout.setSpacing(6)
                    link_hdr = QLabel("Link")
                    link_hdr.setFixedWidth(70)
                    rh_layout.addWidget(link_hdr)
                    fix_hdr = QLabel("Fix")
                    fix_hdr.setFixedWidth(40)
                    rh_layout.addWidget(fix_hdr)
                    hint_hdr = QLabel("Hint")
                    rh_layout.addWidget(hint_hdr)
                    # total right width should accommodate link+fix+hint
                    right_header.setFixedWidth(220)
                    header_h.addWidget(right_header)
                    form_layout.addRow(QLabel(""), header_widget)
                except Exception:
                    pass

                for name, meta in entries:
                    spec_dict = meta
                    val = spec_dict.get("value")
                    ptype = spec_dict.get("type")
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
                            choices = []
                            if spec_dict.get("choices") is not None:
                                choices = list(spec_dict.get("choices"))
                            elif isinstance(val, (list, tuple)):
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
                                elif choices:
                                    w.setCurrentIndex(0)
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

                    hint = ""
                    try:
                        if spec_dict.get("control"):
                            ctrl = spec_dict.get("control") or {}
                            action = ctrl.get("action") or ""
                            mods = ctrl.get("modifiers", []) or []
                            hint = action + ("+" + "+".join(mods) if mods else "")
                            widget.setToolTip(f"Interactive: {hint}")
                    except Exception:
                        hint = ""

                    try:
                        meta_hint = spec_dict.get("hint")
                        if meta_hint:
                            existing_tip = widget.toolTip() if hasattr(widget, "toolTip") else ""
                            combined = f"{existing_tip}\n{meta_hint}".strip()
                            widget.setToolTip(combined)
                    except Exception:
                        pass

                    # Compact sizing for common input widgets (we generally need 3-5 digits)
                    try:
                        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                            widget.setMaximumWidth(80)
                        elif isinstance(widget, QComboBox):
                            widget.setMaximumWidth(140)
                        elif isinstance(widget, QLineEdit):
                            widget.setMaximumWidth(140)
                    except Exception:
                        pass

                    container_h = QWidget()
                    hbox = QHBoxLayout(container_h)
                    hbox.setContentsMargins(0, 0, 0, 0)

                    # Left container holds the main value widget and expands
                    left_container = QWidget()
                    left_l = QHBoxLayout(left_container)
                    left_l.setContentsMargins(0, 0, 0, 0)
                    left_l.addWidget(widget)
                    left_l.addStretch(1)

                    # Link group spinbox: allow user to link parameters together
                    try:
                        link_val = spec_dict.get("link_group", None)
                        link_val = int(link_val) if link_val else 0
                    except Exception:
                        link_val = 0
                    link_spin = QSpinBox()
                    link_spin.setRange(0, 99)
                    link_spin.setValue(link_val)
                    # Compact link control: remove long prefix
                    link_spin.setPrefix("")
                    link_spin.setToolTip("Enter a number to link this parameter with others (0 = not linked)")
                    try:
                        link_spin.blockSignals(True)
                        link_spin.setValue(link_val)
                    finally:
                        link_spin.blockSignals(False)
                    # Apply visual indicator if parameter is linked
                    if link_val > 0:
                        # Use different colors for different link groups
                        link_colors = ["#FFD700", "#87CEEB", "#98FB98", "#FFB6C1", "#DDA0DD", "#F0E68C", "#E6E6FA", "#FFA07A"]
                        color_idx = (link_val - 1) % len(link_colors)
                        link_color = link_colors[color_idx]
                        try:
                            widget.setStyleSheet(f"border: 2px solid {link_color}; border-radius: 3px;")
                        except Exception:
                            pass
                    # bind the link spinbox under a distinct key
                    self._bind_param_widget(f"{name}__link", link_spin)

                    # Fixed checkbox: allow user to mark parameter as fixed during fits
                    try:
                        fixed_val = bool(spec_dict.get("fixed", False))
                    except Exception:
                        fixed_val = False
                    # Shorten label to save horizontal space
                    fixed_chk = QCheckBox("")
                    try:
                        fixed_chk.blockSignals(True)
                        fixed_chk.setChecked(fixed_val)
                    finally:
                        fixed_chk.blockSignals(False)
                    # disable the value widget when fixed to make intent clear
                    try:
                        widget.setEnabled(not fixed_val)
                    except Exception:
                        pass
                    # bind the fixed checkbox under a distinct key so apply/commit sends it
                    self._bind_param_widget(f"{name}__fixed", fixed_chk)

                    # Right container is fixed width so link/fix/hint align across rows
                    right_container = QWidget()
                    right_l = QHBoxLayout(right_container)
                    right_l.setContentsMargins(0, 0, 0, 0)
                    right_l.setSpacing(6)
                    try:
                        link_spin.setFixedWidth(70)
                    except Exception:
                        pass
                    try:
                        fixed_chk.setFixedWidth(40)
                    except Exception:
                        pass
                    right_l.addWidget(link_spin)
                    right_l.addWidget(fixed_chk)
                    # Place interactive hint after link/fix so it's visually grouped to the right
                    if hint:
                        hint_label = QLabel(f"({hint})")
                        hint_label.setStyleSheet("color: gray; font-size: 11px;")
                        hint_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        hint_label.setToolTip(f"Interactive control: {hint}")
                        right_l.addWidget(hint_label)

                    # fix the right container width to match header
                    right_container.setFixedWidth(220)

                    # compose the row: left (expanding) + right (fixed)
                    hbox.addWidget(left_container)
                    hbox.addWidget(right_container)

                    form_layout.addRow(f"{name}:", container_h)
                    self._bind_param_widget(name, widget)

                outer_layout.addWidget(group_box)

            outer_layout.addStretch(1)

            # Preserve current scroll positions so rebuilds don't jump to top
            try:
                vpos = self.param_scroll.verticalScrollBar().value()
            except Exception:
                vpos = 0
            try:
                hpos = self.param_scroll.horizontalScrollBar().value()
            except Exception:
                hpos = 0

            self.param_scroll.takeWidget()
            self.param_scroll.setWidget(new_widget)
            # Restore previous scroll positions where possible
            try:
                self.param_scroll.verticalScrollBar().setValue(vpos)
            except Exception:
                pass
            try:
                self.param_scroll.horizontalScrollBar().setValue(hpos)
            except Exception:
                pass
            self.param_form_widget = new_widget
            self.param_form = None

            control_map = {}
            for name, meta in normalized_specs.items():
                try:
                    ctrl = meta.get("control")
                    if not ctrl:
                        continue
                    action = ctrl.get("action")
                    mods = tuple(sorted(ctrl.get("modifiers", []))) if ctrl.get("modifiers") else tuple()
                    try:
                        step_val = float(meta.get("step", 1.0))
                    except Exception:
                        step_val = 1.0
                    entry = {"name": name, "step": step_val}
                    comp_prefix = meta.get("component")
                    if comp_prefix:
                        entry["component"] = comp_prefix
                    control_map.setdefault((action, mods), []).append(entry)
                except Exception:
                    continue

            # Store the control map for external access (e.g., by InputHandler)
            self._control_map = control_map

        finally:
            self._building_params = False
            # Notify listeners that the parameter widgets have been rebuilt
            try:
                self.parameters_updated.emit()
            except Exception:
                pass

    def _bind_param_widget(self, name: str, widget):
        """
        Bind a parameter widget to track changes.

        Args:
            name: Parameter name
            widget: Widget to bind
        """
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
        """
        Commit a parameter change by emitting a signal.

        Args:
            name: Parameter name
        """
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

        # If this is a fixed-toggle checkbox (named '<param>__fixed'),
        # enable/disable the corresponding value widget for clarity.
        if isinstance(name, str) and name.endswith("__fixed"):
            base = name[: -len("__fixed")]
            try:
                val_widget = self.param_widgets.get(base)
                if val_widget is not None:
                    # disable value widget when fixed
                    try:
                        val_widget.setEnabled(not bool(value))
                    except Exception:
                        pass
            except Exception:
                pass

        self._param_last_values[name] = value
        self.parameter_changed.emit(name, value)

    def get_control_map(self):
        """Get the control map for interactive parameter controls."""
        return getattr(self, '_control_map', {})
