# view/dialogs/save_data_dialog.py
"""
Dialog for saving data, fits, and parameters with multiple output options.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox,
    QDialogButtonBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt
from pathlib import Path


class SaveDataDialog(QDialog):
    """Dialog for selecting save options and file location."""
    
    def __init__(self, parent=None, default_folder=None):
        super().__init__(parent)
        self.setWindowTitle("Save Data and Fits")
        self.setMinimumWidth(500)
        
        self.default_folder = default_folder or str(Path.home())
        self.save_path = None
        self.save_options = {
            'save_image': False,
            'save_ascii': False,
            'save_parameters': False,
            'margin_percent': 10.0,
        }
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Save options group
        options_group = QGroupBox("Save Options")
        options_layout = QVBoxLayout(options_group)
        
        # Radio buttons for save mode
        self.save_mode_group = QButtonGroup(self)
        
        self.save_all_radio = QRadioButton("Save Everything (Image + ASCII + Parameters)")
        self.save_image_radio = QRadioButton("Save as Image Only")
        self.save_ascii_radio = QRadioButton("Save as ASCII Only")
        self.save_params_radio = QRadioButton("Save Parameters Only")
        
        self.save_mode_group.addButton(self.save_all_radio, 0)
        self.save_mode_group.addButton(self.save_image_radio, 1)
        self.save_mode_group.addButton(self.save_ascii_radio, 2)
        self.save_mode_group.addButton(self.save_params_radio, 3)
        
        # Default to save all
        self.save_all_radio.setChecked(True)
        
        options_layout.addWidget(self.save_all_radio)
        options_layout.addWidget(self.save_image_radio)
        options_layout.addWidget(self.save_ascii_radio)
        options_layout.addWidget(self.save_params_radio)
        
        layout.addWidget(options_group)
        
        # Image options group
        image_group = QGroupBox("Image Options")
        image_layout = QHBoxLayout(image_group)
        
        image_layout.addWidget(QLabel("Margin (% beyond data):"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 50.0)
        self.margin_spin.setValue(10.0)
        self.margin_spin.setSuffix("%")
        self.margin_spin.setToolTip("Margin to add beyond the data range (not the fit range)")
        image_layout.addWidget(self.margin_spin)
        image_layout.addStretch()
        
        layout.addWidget(image_group)
        
        # ASCII options info
        ascii_info = QLabel(
            "ASCII format: Data columns (Energy, Counts, Errors) followed by "
            "fine grid fit columns (X_fit, Y_fit_1, Y_fit_2, ...)"
        )
        ascii_info.setWordWrap(True)
        ascii_info.setStyleSheet("QLabel { color: gray; font-size: 9pt; }")
        layout.addWidget(ascii_info)
        
        # Parameter options info
        param_info = QLabel(
            "Parameters format: Tab-separated file with parameter names, values, "
            "and standard errors from fit (if available)"
        )
        param_info.setWordWrap(True)
        param_info.setStyleSheet("QLabel { color: gray; font-size: 9pt; }")
        layout.addWidget(param_info)
        
        # File location group
        location_group = QGroupBox("Save Location")
        location_layout = QVBoxLayout(location_group)
        
        # Base name selection
        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel("Base filename:"))
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_location)
        base_layout.addWidget(self.browse_btn)
        base_layout.addStretch()
        location_layout.addLayout(base_layout)
        
        self.location_label = QLabel("(No location selected)")
        self.location_label.setWordWrap(True)
        self.location_label.setStyleSheet("QLabel { color: gray; font-size: 9pt; }")
        location_layout.addWidget(self.location_label)
        
        layout.addWidget(location_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_save_clicked)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _browse_location(self):
        """Open file dialog to select save location."""
        # Get appropriate file filter based on selected mode
        mode_id = self.save_mode_group.checkedId()
        
        if mode_id == 1:  # Image only
            file_filter = "PNG Image (*.png);;PDF Document (*.pdf)"
            default_ext = ".png"
        elif mode_id == 2:  # ASCII only
            file_filter = "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
            default_ext = ".txt"
        elif mode_id == 3:  # Parameters only
            file_filter = "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
            default_ext = ".txt"
        else:  # Save all
            file_filter = "All Files (*)"
            default_ext = ""
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Save Location",
            self.default_folder,
            file_filter
        )
        
        if filepath:
            # Remove extension if present - we'll add appropriate ones
            p = Path(filepath)
            if p.suffix:
                base_path = str(p.parent / p.stem)
            else:
                base_path = filepath
            
            self.save_path = base_path
            self.location_label.setText(f"Base: {self.save_path}")
    
    def _on_save_clicked(self):
        """Validate and accept the dialog."""
        if not self.save_path:
            QMessageBox.warning(
                self,
                "No Location Selected",
                "Please select a save location using the Browse button."
            )
            return
        
        # Collect save options
        mode_id = self.save_mode_group.checkedId()
        
        if mode_id == 0:  # Save all
            self.save_options['save_image'] = True
            self.save_options['save_ascii'] = True
            self.save_options['save_parameters'] = True
        elif mode_id == 1:  # Image only
            self.save_options['save_image'] = True
            self.save_options['save_ascii'] = False
            self.save_options['save_parameters'] = False
        elif mode_id == 2:  # ASCII only
            self.save_options['save_image'] = False
            self.save_options['save_ascii'] = True
            self.save_options['save_parameters'] = False
        elif mode_id == 3:  # Parameters only
            self.save_options['save_image'] = False
            self.save_options['save_ascii'] = False
            self.save_options['save_parameters'] = True
        
        self.save_options['margin_percent'] = self.margin_spin.value()
        
        self.accept()
    
    def get_save_path(self):
        """Get the base save path (without extension)."""
        return self.save_path
    
    def get_save_options(self):
        """Get the dict of save options."""
        return self.save_options
