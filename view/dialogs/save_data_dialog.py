# view/dialogs/save_data_dialog.py
"""
Dialog for saving data, fits, and parameters with multiple output options.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox,
    QDialogButtonBox, QRadioButton, QButtonGroup, QLineEdit, QComboBox
)
from PySide6.QtCore import Qt
from pathlib import Path


class SaveDataDialog(QDialog):
    """Dialog for selecting save options and file location."""
    
    def __init__(self, parent=None, default_folder=None, default_filename=None):
        super().__init__(parent)
        self.setWindowTitle("Save Data and Fits")
        self.setMinimumWidth(600)
        
        self.default_folder = default_folder or str(Path.home())
        self.default_filename = default_filename or "data"
        self.save_options = {
            'save_image': False,
            'save_ascii': False,
            'save_parameters': False,
            'margin_percent': 10.0,
            'delimiter': 'comma',
        }
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # File location group (at top now)
        location_group = QGroupBox("Save Location")
        location_layout = QVBoxLayout(location_group)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_edit = QLineEdit(self.default_folder)
        self.folder_edit.setReadOnly(True)
        folder_layout.addWidget(self.folder_edit, 1)
        self.browse_folder_btn = QPushButton("Browse...")
        self.browse_folder_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.browse_folder_btn)
        location_layout.addLayout(folder_layout)
        
        # Filename input
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Base filename:"))
        self.filename_edit = QLineEdit(self.default_filename)
        self.filename_edit.setPlaceholderText("Enter base filename (no extension)")
        filename_layout.addWidget(self.filename_edit, 1)
        location_layout.addLayout(filename_layout)
        
        layout.addWidget(location_group)
        
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
        
        # ASCII options group
        ascii_group = QGroupBox("ASCII Options")
        ascii_layout = QHBoxLayout(ascii_group)
        
        ascii_layout.addWidget(QLabel("Delimiter:"))
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems(["Comma", "Tab", "Space"])
        self.delimiter_combo.setCurrentIndex(0)  # Default to comma
        ascii_layout.addWidget(self.delimiter_combo)
        ascii_layout.addStretch()
        
        layout.addWidget(ascii_group)
        
        # Info labels
        info_label = QLabel(
            "Files will be saved with suffixes: _image.png, _ASCII.txt, _params.txt"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; font-size: 9pt; }")
        layout.addWidget(info_label)
        
        # Save action buttons (one for each type)
        buttons_group = QGroupBox("Save Actions")
        buttons_layout = QVBoxLayout(buttons_group)
        
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self._save_image)
        buttons_layout.addWidget(self.save_image_btn)
        
        self.save_ascii_btn = QPushButton("Save ASCII Data")
        self.save_ascii_btn.clicked.connect(self._save_ascii)
        buttons_layout.addWidget(self.save_ascii_btn)
        
        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.clicked.connect(self._save_params)
        buttons_layout.addWidget(self.save_params_btn)
        
        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.clicked.connect(self._save_all)
        self.save_all_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        buttons_layout.addWidget(self.save_all_btn)
        
        layout.addWidget(buttons_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: green; font-size: 9pt; }")
        layout.addWidget(self.status_label)
        
        # Close button at bottom
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        close_layout.addWidget(self.close_btn)
        layout.addLayout(close_layout)
    
    def _browse_folder(self):
        """Open folder dialog to select save folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            self.folder_edit.text()
        )
        
        if folder:
            self.folder_edit.setText(folder)
    
    def _get_base_path(self):
        """Get the full base path from folder and filename."""
        folder = self.folder_edit.text().strip()
        filename = self.filename_edit.text().strip()
        
        if not folder or not filename:
            return None
        
        return str(Path(folder) / filename)
    
    def _validate_inputs(self):
        """Validate that folder and filename are provided."""
        if not self.folder_edit.text().strip():
            QMessageBox.warning(
                self,
                "No Folder Selected",
                "Please select a save folder."
            )
            return False
        
        if not self.filename_edit.text().strip():
            QMessageBox.warning(
                self,
                "No Filename Provided",
                "Please enter a base filename."
            )
            return False
        
        return True
    
    def _get_delimiter(self):
        """Get the selected delimiter character."""
        delimiter_text = self.delimiter_combo.currentText()
        if delimiter_text == "Comma":
            return ","
        elif delimiter_text == "Tab":
            return "\t"
        elif delimiter_text == "Space":
            return " "
        return ","
    
    def _save_image(self):
        """Save image only."""
        if not self._validate_inputs():
            return
        
        self.save_options['save_image'] = True
        self.save_options['save_ascii'] = False
        self.save_options['save_parameters'] = False
        self.save_options['margin_percent'] = self.margin_spin.value()
        self.save_options['delimiter'] = self._get_delimiter()
        
        # Signal to parent that save was requested (don't close dialog)
        self.status_label.setText("Saving image...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
    
    def _save_ascii(self):
        """Save ASCII data only."""
        if not self._validate_inputs():
            return
        
        self.save_options['save_image'] = False
        self.save_options['save_ascii'] = True
        self.save_options['save_parameters'] = False
        self.save_options['margin_percent'] = self.margin_spin.value()
        self.save_options['delimiter'] = self._get_delimiter()
        
        self.status_label.setText("Saving ASCII data...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
    
    def _save_params(self):
        """Save parameters only."""
        if not self._validate_inputs():
            return
        
        self.save_options['save_image'] = False
        self.save_options['save_ascii'] = False
        self.save_options['save_parameters'] = True
        self.save_options['margin_percent'] = self.margin_spin.value()
        self.save_options['delimiter'] = self._get_delimiter()
        
        self.status_label.setText("Saving parameters...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
    
    def _save_all(self):
        """Save everything."""
        if not self._validate_inputs():
            return
        
        self.save_options['save_image'] = True
        self.save_options['save_ascii'] = True
        self.save_options['save_parameters'] = True
        self.save_options['margin_percent'] = self.margin_spin.value()
        self.save_options['delimiter'] = self._get_delimiter()
        
        self.status_label.setText("Saving all formats...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
    
    def get_base_path(self):
        """Get the full base save path (without extension)."""
        return self._get_base_path()
    
    def get_save_options(self):
        """Get the dict of save options."""
        return self.save_options
    
    def set_status(self, message, is_success=True):
        """Set status message with appropriate styling."""
        self.status_label.setText(message)
        if is_success:
            self.status_label.setStyleSheet("QLabel { color: green; font-size: 9pt; }")
        else:
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 9pt; }")
    
    def get_delimiter_name(self):
        """Get the delimiter name for config storage."""
        delimiter_text = self.delimiter_combo.currentText().lower()
        return delimiter_text
    
    def set_delimiter_from_config(self, delimiter_name):
        """Set the delimiter combo from config value."""
        if delimiter_name == "comma":
            self.delimiter_combo.setCurrentIndex(0)
        elif delimiter_name == "tab":
            self.delimiter_combo.setCurrentIndex(1)
        elif delimiter_name == "space":
            self.delimiter_combo.setCurrentIndex(2)
