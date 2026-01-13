"""
Instrument dock widget for PUFFIN.

Provides controls for instrument configuration including slits, collimators,
crystals, focusing options, and experimental modules.
"""
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox, QGroupBox,
    QScrollArea, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, Optional
from dataio.instrument_config import InstrumentConfig


class InstrumentDock(QDockWidget):
    """Dock widget for instrument configuration and control."""
    
    # Signals
    instrument_selected = Signal(str)  # instrument name
    slit_changed = Signal(str, str, float)  # position_name, dimension_name, value
    collimator_changed = Signal(str, object)  # position_name, value
    crystal_changed = Signal(str, str)  # component (mono/analyzer), crystal_name
    focusing_changed = Signal(str, str)  # component (mono/analyzer), focusing_type
    module_enabled_changed = Signal(str, bool)  # module_name, enabled
    module_parameter_changed = Signal(str, str, object)  # module_name, param_name, value
    
    def __init__(self, parent=None):
        super().__init__("Instrument", parent)
        self.setObjectName("InstrumentDock")
        
        # Current instrument configuration
        self._current_config: Optional[InstrumentConfig] = None
        
        # Widget storage
        self._slit_widgets: Dict[str, Dict[str, QDoubleSpinBox]] = {}
        self._collimator_widgets: Dict[str, QComboBox] = {}
        self._crystal_widgets: Dict[str, QComboBox] = {}
        self._focusing_widgets: Dict[str, QComboBox] = {}
        self._module_widgets: Dict[str, Dict[str, Any]] = {}
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Instrument selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        self.instrument_combo.currentTextChanged.connect(self._on_instrument_selected)
        selector_layout.addWidget(self.instrument_combo, 1)
        
        self.load_instrument_btn = QPushButton("Load")
        self.load_instrument_btn.clicked.connect(self._on_load_instrument_clicked)
        selector_layout.addWidget(self.load_instrument_btn)
        
        main_layout.addLayout(selector_layout)
        
        # Scrollable area for instrument controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)
        self.controls_layout.setSpacing(8)
        
        scroll.setWidget(self.controls_widget)
        main_layout.addWidget(scroll, 1)
        
        self.setWidget(main_widget)
    
    def populate_instruments(self, instruments: list):
        """
        Populate the instrument selector with available instruments.
        
        Args:
            instruments: List of dicts with 'name', 'path', 'description' keys
        """
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        
        for inst in instruments:
            name = inst.get('name', 'Unknown')
            self.instrument_combo.addItem(name, userData=inst)
        
        self.instrument_combo.blockSignals(False)
        
        # Select first instrument if available
        if self.instrument_combo.count() > 0:
            self.instrument_combo.setCurrentIndex(0)
    
    def set_instrument_config(self, config: InstrumentConfig):
        """
        Set the current instrument configuration and rebuild controls.
        
        Args:
            config: InstrumentConfig object
        """
        self._current_config = config
        self._rebuild_controls()
    
    def _rebuild_controls(self):
        """Rebuild all instrument controls based on current configuration."""
        # Clear existing controls
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._slit_widgets.clear()
        self._collimator_widgets.clear()
        self._crystal_widgets.clear()
        self._focusing_widgets.clear()
        self._module_widgets.clear()
        
        if not self._current_config:
            self.controls_layout.addWidget(QLabel("No instrument loaded"))
            self.controls_layout.addStretch()
            return
        
        # Build sections
        self._build_crystal_section()
        self._build_focusing_section()
        self._build_collimator_section()
        self._build_slit_section()
        self._build_module_section()
        
        self.controls_layout.addStretch()
    
    def _build_crystal_section(self):
        """Build crystal selection controls."""
        if not self._current_config:
            return
        
        mono_crystals = self._current_config.get_monochromator_crystals()
        analyzer_crystals = self._current_config.get_analyzer_crystals()
        
        if not mono_crystals and not analyzer_crystals:
            return
        
        group = QGroupBox("Crystals")
        form = QFormLayout(group)
        
        if mono_crystals:
            mono_combo = QComboBox()
            for crystal in mono_crystals:
                mono_combo.addItem(f"{crystal.name} (d={crystal.d_spacing:.3f}Å)")
            mono_combo.currentTextChanged.connect(
                lambda text: self._on_crystal_changed("monochromator", text)
            )
            form.addRow("Monochromator:", mono_combo)
            self._crystal_widgets["monochromator"] = mono_combo
        
        if analyzer_crystals:
            analyzer_combo = QComboBox()
            for crystal in analyzer_crystals:
                analyzer_combo.addItem(f"{crystal.name} (d={crystal.d_spacing:.3f}Å)")
            analyzer_combo.currentTextChanged.connect(
                lambda text: self._on_crystal_changed("analyzer", text)
            )
            form.addRow("Analyzer:", analyzer_combo)
            self._crystal_widgets["analyzer"] = analyzer_combo
        
        self.controls_layout.addWidget(group)
    
    def _build_focusing_section(self):
        """Build focusing option controls."""
        if not self._current_config:
            return
        
        mono_focusing = self._current_config.get_monochromator_focusing_options()
        analyzer_focusing = self._current_config.get_analyzer_focusing_options()
        
        if not mono_focusing and not analyzer_focusing:
            return
        
        group = QGroupBox("Focusing")
        form = QFormLayout(group)
        
        if mono_focusing:
            mono_combo = QComboBox()
            mono_combo.addItems(mono_focusing)
            mono_combo.currentTextChanged.connect(
                lambda text: self._on_focusing_changed("monochromator", text)
            )
            form.addRow("Monochromator:", mono_combo)
            self._focusing_widgets["monochromator"] = mono_combo
        
        if analyzer_focusing:
            analyzer_combo = QComboBox()
            analyzer_combo.addItems(analyzer_focusing)
            analyzer_combo.currentTextChanged.connect(
                lambda text: self._on_focusing_changed("analyzer", text)
            )
            form.addRow("Analyzer:", analyzer_combo)
            self._focusing_widgets["analyzer"] = analyzer_combo
        
        self.controls_layout.addWidget(group)
    
    def _build_collimator_section(self):
        """Build collimator controls."""
        if not self._current_config or not self._current_config.collimators:
            return
        
        group = QGroupBox("Collimators")
        form = QFormLayout(group)
        
        for collimator in self._current_config.collimators:
            combo = QComboBox()
            for option in collimator.options:
                combo.addItem(str(option))
            
            # Connect with position name
            combo.currentTextChanged.connect(
                lambda text, name=collimator.name: self._on_collimator_changed(name, text)
            )
            
            form.addRow(f"{collimator.label}:", combo)
            self._collimator_widgets[collimator.name] = combo
        
        self.controls_layout.addWidget(group)
    
    def _build_slit_section(self):
        """Build slit controls."""
        if not self._current_config or not self._current_config.slits:
            return
        
        group = QGroupBox("Slits")
        layout = QVBoxLayout(group)
        
        for slit_pos in self._current_config.slits:
            # Create subgroup for this slit position
            pos_group = QGroupBox(slit_pos.label)
            pos_form = QFormLayout(pos_group)
            
            pos_widgets = {}
            for dim in slit_pos.dimensions:
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(dim.min)
                spinbox.setMaximum(dim.max)
                spinbox.setValue(dim.default)
                spinbox.setSingleStep(dim.step)
                spinbox.setDecimals(1)
                spinbox.setSuffix(" mm")
                
                # Connect with position and dimension names
                spinbox.valueChanged.connect(
                    lambda val, pos=slit_pos.name, dim_name=dim.name: 
                    self._on_slit_changed(pos, dim_name, val)
                )
                
                pos_form.addRow(f"{dim.label}:", spinbox)
                pos_widgets[dim.name] = spinbox
            
            self._slit_widgets[slit_pos.name] = pos_widgets
            layout.addWidget(pos_group)
        
        self.controls_layout.addWidget(group)
    
    def _build_module_section(self):
        """Build experimental module controls."""
        if not self._current_config or not self._current_config.modules:
            return
        
        group = QGroupBox("Experimental Modules")
        layout = QVBoxLayout(group)
        
        for module in self._current_config.modules:
            # Create subgroup for this module
            module_group = QGroupBox(module.label)
            module_layout = QVBoxLayout(module_group)
            
            # Enable checkbox
            enable_check = QCheckBox("Enable")
            enable_check.setChecked(module.enabled)
            enable_check.stateChanged.connect(
                lambda state, name=module.name: 
                self._on_module_enabled_changed(name, bool(state))
            )
            module_layout.addWidget(enable_check)
            
            # Parameters form
            if module.parameters:
                param_form = QFormLayout()
                param_widgets = {}
                
                for param in module.parameters:
                    widget = self._create_parameter_widget(param)
                    if widget:
                        # Connect parameter changes
                        self._connect_parameter_widget(
                            widget, param, module.name
                        )
                        param_form.addRow(f"{param.label}:", widget)
                        param_widgets[param.name] = widget
                
                module_layout.addLayout(param_form)
                
                self._module_widgets[module.name] = {
                    'enable': enable_check,
                    'params': param_widgets
                }
            
            layout.addWidget(module_group)
        
        self.controls_layout.addWidget(group)
    
    def _create_parameter_widget(self, param):
        """Create appropriate widget for a module parameter."""
        if param.type in ("float", "double"):
            widget = QDoubleSpinBox()
            if param.min is not None:
                widget.setMinimum(param.min)
            if param.max is not None:
                widget.setMaximum(param.max)
            if param.default is not None:
                widget.setValue(float(param.default))
            return widget
        elif param.type == "int":
            from PySide6.QtWidgets import QSpinBox
            widget = QSpinBox()
            if param.min is not None:
                widget.setMinimum(int(param.min))
            if param.max is not None:
                widget.setMaximum(int(param.max))
            if param.default is not None:
                widget.setValue(int(param.default))
            return widget
        elif param.type == "bool":
            widget = QCheckBox()
            if param.default is not None:
                widget.setChecked(bool(param.default))
            return widget
        elif param.type == "choice" and param.choices:
            widget = QComboBox()
            widget.addItems(param.choices)
            if param.default in param.choices:
                widget.setCurrentText(str(param.default))
            return widget
        else:
            from PySide6.QtWidgets import QLineEdit
            widget = QLineEdit()
            if param.default is not None:
                widget.setText(str(param.default))
            return widget
    
    def _connect_parameter_widget(self, widget, param, module_name):
        """Connect parameter widget signals."""
        if isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(
                lambda val: self._on_module_parameter_changed(
                    module_name, param.name, val
                )
            )
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(
                lambda state: self._on_module_parameter_changed(
                    module_name, param.name, bool(state)
                )
            )
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(
                lambda text: self._on_module_parameter_changed(
                    module_name, param.name, text
                )
            )
    
    def _on_instrument_selected(self, name: str):
        """Handle instrument selection from combo box."""
        if name:
            self.instrument_selected.emit(name)
    
    def _on_load_instrument_clicked(self):
        """Handle load instrument button click."""
        idx = self.instrument_combo.currentIndex()
        if idx >= 0:
            data = self.instrument_combo.itemData(idx)
            if data:
                name = data.get('name', '')
                if name:
                    self.instrument_selected.emit(name)
    
    def _on_slit_changed(self, position: str, dimension: str, value: float):
        """Handle slit value change."""
        self.slit_changed.emit(position, dimension, value)
    
    def _on_collimator_changed(self, position: str, value: str):
        """Handle collimator selection change."""
        # Try to convert to number if possible
        try:
            numeric_value = float(value)
            self.collimator_changed.emit(position, numeric_value)
        except ValueError:
            self.collimator_changed.emit(position, value)
    
    def _on_crystal_changed(self, component: str, text: str):
        """Handle crystal selection change."""
        # Extract crystal name from text (before the d_spacing info)
        crystal_name = text.split('(')[0].strip() if '(' in text else text
        self.crystal_changed.emit(component, crystal_name)
    
    def _on_focusing_changed(self, component: str, focusing_type: str):
        """Handle focusing option change."""
        self.focusing_changed.emit(component, focusing_type)
    
    def _on_module_enabled_changed(self, module_name: str, enabled: bool):
        """Handle module enable/disable."""
        self.module_enabled_changed.emit(module_name, enabled)
    
    def _on_module_parameter_changed(self, module_name: str, param_name: str, value):
        """Handle module parameter change."""
        self.module_parameter_changed.emit(module_name, param_name, value)
    
    def get_slit_values(self) -> Dict[str, Dict[str, float]]:
        """Get current values of all slits."""
        values = {}
        for pos_name, widgets in self._slit_widgets.items():
            values[pos_name] = {
                dim_name: widget.value()
                for dim_name, widget in widgets.items()
            }
        return values
    
    def get_collimator_values(self) -> Dict[str, Any]:
        """Get current values of all collimators."""
        values = {}
        for pos_name, widget in self._collimator_widgets.items():
            text = widget.currentText()
            try:
                values[pos_name] = float(text)
            except ValueError:
                values[pos_name] = text
        return values
    
    def get_crystal_selections(self) -> Dict[str, str]:
        """Get current crystal selections."""
        selections = {}
        for component, widget in self._crystal_widgets.items():
            text = widget.currentText()
            crystal_name = text.split('(')[0].strip() if '(' in text else text
            selections[component] = crystal_name
        return selections
    
    def get_focusing_selections(self) -> Dict[str, str]:
        """Get current focusing selections."""
        return {
            component: widget.currentText()
            for component, widget in self._focusing_widgets.items()
        }
