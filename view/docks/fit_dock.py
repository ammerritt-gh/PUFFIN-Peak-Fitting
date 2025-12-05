"""
Fit dock widget for advanced fitting options.

This dock provides controls for fitting with options like:
- Fit one step (single iteration)
- Fit 10 steps (10 iterations)
- Fit to completion (until convergence)
- Parameter bounds (min/max limits)
- Revert to previous parameters
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QScrollArea, QSpinBox, QCheckBox,
    QLineEdit, QHBoxLayout, QGroupBox, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, Signal
import math
from functools import partial


# Constants
FLOAT_COMPARISON_TOLERANCE = 1e-9  # Tolerance for float equality comparisons


class FitDock(QDockWidget):
    """Floating dock widget for advanced fitting options."""

    # Signals
    fit_one_step_clicked = Signal()
    fit_n_steps_clicked = Signal(int)  # number of steps
    fit_to_completion_clicked = Signal()
    revert_clicked = Signal()
    bounds_changed = Signal(str, object, object)  # param_name, min_val, max_val

    def __init__(self, parent=None):
        """
        Initialize the fit dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Fit Settings", parent)
        self._param_bound_widgets = {}  # name -> (min_widget, max_widget)
        self._building_params = False
        self._fit_in_progress = False
        
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
        vlayout.setSpacing(10)

        # Fit options group
        fit_group = QGroupBox("Fit Options")
        fit_layout = QVBoxLayout(fit_group)
        fit_layout.setSpacing(8)

        # Fit one step button
        self.fit_one_step_btn = QPushButton("Fit One Step")
        self.fit_one_step_btn.setToolTip("Run one iteration of the fitting process")
        fit_layout.addWidget(self.fit_one_step_btn)

        # Fit N steps with spinbox
        steps_row = QHBoxLayout()
        self.fit_n_steps_btn = QPushButton("Fit Steps:")
        self.fit_n_steps_btn.setToolTip("Run N iterations of the fitting process")
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 1000)
        self.steps_spinbox.setValue(10)
        self.steps_spinbox.setToolTip("Number of fitting iterations")
        steps_row.addWidget(self.fit_n_steps_btn)
        steps_row.addWidget(self.steps_spinbox)
        steps_row.addStretch(1)
        fit_layout.addLayout(steps_row)

        # Fit to completion button
        self.fit_to_completion_btn = QPushButton("Fit to Completion")
        self.fit_to_completion_btn.setToolTip("Run fitting until convergence or max iterations")
        fit_layout.addWidget(self.fit_to_completion_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        fit_layout.addWidget(self.progress_bar)

        vlayout.addWidget(fit_group)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        vlayout.addWidget(separator)

        # Revert group
        revert_group = QGroupBox("Undo Fit")
        revert_layout = QVBoxLayout(revert_group)
        
        self.revert_btn = QPushButton("Revert to Previous")
        self.revert_btn.setToolTip("Restore parameters to values before fitting")
        self.revert_btn.setEnabled(False)  # Disabled until a fit is performed
        revert_layout.addWidget(self.revert_btn)
        
        self.revert_status_label = QLabel("No previous fit to revert to")
        self.revert_status_label.setStyleSheet("color: gray; font-style: italic;")
        revert_layout.addWidget(self.revert_status_label)

        vlayout.addWidget(revert_group)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        vlayout.addWidget(separator2)

        # Parameter bounds section
        bounds_label = QLabel("Parameter Bounds")
        bounds_label.setStyleSheet("font-weight: bold;")
        vlayout.addWidget(bounds_label)

        # Scrollable bounds area
        self.bounds_scroll = QScrollArea()
        self.bounds_scroll.setWidgetResizable(True)
        self.bounds_form_widget = QWidget()
        self.bounds_form = QFormLayout(self.bounds_form_widget)
        self.bounds_form.setContentsMargins(6, 6, 6, 6)
        self.bounds_form.setSpacing(6)
        self.bounds_scroll.setWidget(self.bounds_form_widget)
        vlayout.addWidget(self.bounds_scroll)

        vlayout.addStretch(1)

        self.setWidget(container)
        
        # Reasonable default size
        self.setMinimumWidth(350)
        self.resize(380, 500)

        # Connect signals
        self.fit_one_step_btn.clicked.connect(self._on_fit_one_step)
        self.fit_n_steps_btn.clicked.connect(self._on_fit_n_steps)
        self.fit_to_completion_btn.clicked.connect(self._on_fit_to_completion)
        self.revert_btn.clicked.connect(self._on_revert)

    def _on_fit_one_step(self):
        """Handle fit one step button click."""
        self.fit_one_step_clicked.emit()

    def _on_fit_n_steps(self):
        """Handle fit N steps button click."""
        n = self.steps_spinbox.value()
        self.fit_n_steps_clicked.emit(n)

    def _on_fit_to_completion(self):
        """Handle fit to completion button click."""
        self.fit_to_completion_clicked.emit()

    def _on_revert(self):
        """Handle revert button click."""
        self.revert_clicked.emit()

    def set_fit_in_progress(self, in_progress: bool):
        """Update UI to reflect fit status."""
        self._fit_in_progress = in_progress
        self.fit_one_step_btn.setEnabled(not in_progress)
        self.fit_n_steps_btn.setEnabled(not in_progress)
        self.fit_to_completion_btn.setEnabled(not in_progress)
        self.steps_spinbox.setEnabled(not in_progress)
        self.progress_bar.setVisible(in_progress)
        if not in_progress:
            self.progress_bar.setValue(0)

    def update_progress(self, progress: float):
        """Update the progress bar.
        
        Args:
            progress: Progress value from 0.0 to 1.0
        """
        self.progress_bar.setValue(int(progress * 100))

    def set_revert_available(self, available: bool, message: str = None):
        """Enable/disable the revert button.
        
        Args:
            available: Whether revert is available
            message: Optional status message
        """
        self.revert_btn.setEnabled(available)
        if message:
            self.revert_status_label.setText(message)
        elif available:
            self.revert_status_label.setText("Previous fit available for revert")
            self.revert_status_label.setStyleSheet("color: green;")
        else:
            self.revert_status_label.setText("No previous fit to revert to")
            self.revert_status_label.setStyleSheet("color: gray; font-style: italic;")

    def populate_bounds(self, param_specs: dict):
        """
        Populate the bounds form with parameter min/max fields.

        Args:
            param_specs: Dict of parameter specifications
        """
        self._building_params = True
        try:
            # Create new widget for bounds
            new_widget = QWidget()
            form_layout = QFormLayout(new_widget)
            form_layout.setContentsMargins(6, 6, 6, 6)
            form_layout.setSpacing(6)

            self._param_bound_widgets = {}

            if not param_specs:
                label = QLabel("No parameters available.\nLoad a model to set bounds.")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("color: gray; font-style: italic;")
                form_layout.addRow(label)
            else:
                for name, meta in param_specs.items():
                    if isinstance(meta, dict):
                        spec_dict = meta
                    else:
                        spec_dict = {"value": meta}

                    current_min = spec_dict.get("min")
                    current_max = spec_dict.get("max")
                    
                    # Skip fixed parameters
                    if spec_dict.get("fixed", False):
                        continue

                    # Create min/max row
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(4)

                    # Min value
                    min_edit = QLineEdit()
                    min_edit.setPlaceholderText("Min")
                    min_edit.setMaximumWidth(80)
                    if current_min is not None and current_min > -1e9:
                        min_edit.setText(str(current_min))
                    min_edit.setToolTip(f"Minimum value for {name} (leave blank for no limit)")
                    
                    # Max value
                    max_edit = QLineEdit()
                    max_edit.setPlaceholderText("Max")
                    max_edit.setMaximumWidth(80)
                    if current_max is not None and current_max < 1e9:
                        max_edit.setText(str(current_max))
                    max_edit.setToolTip(f"Maximum value for {name} (leave blank for no limit)")

                    row_layout.addWidget(QLabel("Min:"))
                    row_layout.addWidget(min_edit)
                    row_layout.addWidget(QLabel("Max:"))
                    row_layout.addWidget(max_edit)
                    row_layout.addStretch(1)

                    form_layout.addRow(f"{name}:", row_widget)
                    
                    self._param_bound_widgets[name] = (min_edit, max_edit)
                    
                    # Connect signals
                    min_edit.editingFinished.connect(
                        partial(self._on_bound_changed, name)
                    )
                    max_edit.editingFinished.connect(
                        partial(self._on_bound_changed, name)
                    )

            # Replace scroll content
            self.bounds_scroll.takeWidget()
            self.bounds_scroll.setWidget(new_widget)
            self.bounds_form_widget = new_widget
            self.bounds_form = form_layout

        finally:
            self._building_params = False

    def _on_bound_changed(self, name: str):
        """Handle bound value change."""
        if self._building_params:
            return
        
        widgets = self._param_bound_widgets.get(name)
        if widgets is None:
            return
        
        min_edit, max_edit = widgets
        min_val = None
        max_val = None
        
        # Parse min value
        min_text = min_edit.text().strip()
        if min_text:
            try:
                min_val = float(min_text)
            except ValueError:
                pass
        
        # Parse max value
        max_text = max_edit.text().strip()
        if max_text:
            try:
                max_val = float(max_text)
            except ValueError:
                pass
        
        self.bounds_changed.emit(name, min_val, max_val)

    def get_bounds(self) -> dict:
        """
        Get the current bounds settings.

        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        bounds = {}
        for name, (min_edit, max_edit) in self._param_bound_widgets.items():
            min_val = None
            max_val = None
            
            min_text = min_edit.text().strip()
            if min_text:
                try:
                    min_val = float(min_text)
                except ValueError:
                    pass
            
            max_text = max_edit.text().strip()
            if max_text:
                try:
                    max_val = float(max_text)
                except ValueError:
                    pass
            
            bounds[name] = (min_val, max_val)
        
        return bounds

    def update_bound_values(self, name: str, min_val, max_val):
        """Update the bound widgets for a specific parameter.
        
        Args:
            name: Parameter name
            min_val: Minimum value or None
            max_val: Maximum value or None
        """
        widgets = self._param_bound_widgets.get(name)
        if widgets is None:
            return
        
        min_edit, max_edit = widgets
        
        try:
            min_edit.blockSignals(True)
            max_edit.blockSignals(True)
            
            if min_val is not None and min_val > -1e9:
                min_edit.setText(str(min_val))
            else:
                min_edit.setText("")
            
            if max_val is not None and max_val < 1e9:
                max_edit.setText(str(max_val))
            else:
                max_edit.setText("")
        finally:
            min_edit.blockSignals(False)
            max_edit.blockSignals(False)
