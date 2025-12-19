"""
Save dock widget for exporting data, fits, and parameters.
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QDoubleSpinBox, QMessageBox, QLineEdit, QComboBox,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path


class SaveDock(QDockWidget):
    """Dock widget for saving data with multiple export options."""
    
    # Signals for save operations
    save_image_with_margin_requested = Signal(dict)  # options dict
    save_image_view_range_requested = Signal(dict)  # options dict
    save_ascii_requested = Signal(dict)  # options dict
    save_parameters_requested = Signal(dict)  # options dict
    save_all_requested = Signal(dict)  # options dict
    
    def __init__(self, parent=None, default_folder=None, default_filename=None):
        """
        Initialize the save dock.
        
        Args:
            parent: Parent widget (typically the main window)
            default_folder: Default folder for saving files
            default_filename: Default base filename
        """
        super().__init__("Save Data", parent)
        self.default_folder = default_folder or str(Path.home())
        self.default_filename = default_filename or "data"
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Main widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # File location group
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
        
        # Base filename input
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Base:"))
        self.base_filename_edit = QLineEdit(self.default_filename)
        self.base_filename_edit.setPlaceholderText("Enter base filename")
        self.base_filename_edit.textChanged.connect(self._update_preview_filenames)
        filename_layout.addWidget(self.base_filename_edit, 1)
        location_layout.addLayout(filename_layout)
        
        layout.addWidget(location_group)
        
        # Image options group
        image_group = QGroupBox("Image Options")
        image_layout = QVBoxLayout(image_group)
        
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Margin %:"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 50.0)
        self.margin_spin.setValue(10.0)
        self.margin_spin.setSuffix("%")
        self.margin_spin.setToolTip("Margin beyond data range")
        margin_layout.addWidget(self.margin_spin)
        margin_layout.addStretch()
        image_layout.addLayout(margin_layout)
        
        layout.addWidget(image_group)
        
        # ASCII options group
        ascii_group = QGroupBox("ASCII Options")
        ascii_layout = QHBoxLayout(ascii_group)
        
        ascii_layout.addWidget(QLabel("Delimiter:"))
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems(["Comma", "Tab", "Space"])
        self.delimiter_combo.setCurrentIndex(0)
        ascii_layout.addWidget(self.delimiter_combo)
        ascii_layout.addStretch()
        
        layout.addWidget(ascii_group)
        
        # Save actions group with filename previews
        actions_group = QGroupBox("Save Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Image with margin
        img_margin_layout = QHBoxLayout()
        self.save_image_margin_btn = QPushButton("Image (Margin)")
        self.save_image_margin_btn.setToolTip("Save image with margin around data")
        self.save_image_margin_btn.clicked.connect(self._save_image_margin)
        img_margin_layout.addWidget(self.save_image_margin_btn)
        self.image_margin_filename = QLineEdit()
        self.image_margin_filename.setPlaceholderText("filename_image_margin.png")
        img_margin_layout.addWidget(self.image_margin_filename, 1)
        actions_layout.addLayout(img_margin_layout)
        
        # Image with view range
        img_view_layout = QHBoxLayout()
        self.save_image_view_btn = QPushButton("Image (View)")
        self.save_image_view_btn.setToolTip("Save image using current plot view range")
        self.save_image_view_btn.clicked.connect(self._save_image_view)
        img_view_layout.addWidget(self.save_image_view_btn)
        self.image_view_filename = QLineEdit()
        self.image_view_filename.setPlaceholderText("filename_image_view.png")
        img_view_layout.addWidget(self.image_view_filename, 1)
        actions_layout.addLayout(img_view_layout)
        
        # ASCII data
        ascii_layout = QHBoxLayout()
        self.save_ascii_btn = QPushButton("ASCII Data")
        self.save_ascii_btn.clicked.connect(self._save_ascii)
        ascii_layout.addWidget(self.save_ascii_btn)
        self.ascii_filename = QLineEdit()
        self.ascii_filename.setPlaceholderText("filename_ASCII.txt")
        ascii_layout.addWidget(self.ascii_filename, 1)
        actions_layout.addLayout(ascii_layout)
        
        # Parameters
        params_layout = QHBoxLayout()
        self.save_params_btn = QPushButton("Parameters")
        self.save_params_btn.clicked.connect(self._save_params)
        params_layout.addWidget(self.save_params_btn)
        self.params_filename = QLineEdit()
        self.params_filename.setPlaceholderText("filename_params.txt")
        params_layout.addWidget(self.params_filename, 1)
        actions_layout.addLayout(params_layout)
        
        # Save all
        all_layout = QHBoxLayout()
        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self.save_all_btn.clicked.connect(self._save_all)
        all_layout.addWidget(self.save_all_btn)
        all_layout.addStretch()
        actions_layout.addLayout(all_layout)
        
        layout.addWidget(actions_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: green; font-size: 9pt; }")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Update initial preview filenames
        self._update_preview_filenames()
        
        scroll_area.setWidget(main_widget)
        self.setWidget(scroll_area)
    
    def _browse_folder(self):
        """Open folder dialog to select save folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            self.folder_edit.text()
        )
        
        if folder:
            self.folder_edit.setText(folder)
    
    def _update_preview_filenames(self):
        """Update the preview filename fields based on base filename."""
        base = self.base_filename_edit.text().strip() or "data"
        
        # Update each filename preview
        self.image_margin_filename.setText(f"{base}_image_margin.png")
        self.image_view_filename.setText(f"{base}_image_view.png")
        self.ascii_filename.setText(f"{base}_ASCII.txt")
        self.params_filename.setText(f"{base}_params.txt")
    
    def _get_save_path(self, suffix):
        """Get the full save path for a given suffix."""
        folder = self.folder_edit.text().strip()
        base = self.base_filename_edit.text().strip()
        
        if not folder or not base:
            return None
        
        return str(Path(folder) / f"{base}{suffix}")
    
    def _validate_inputs(self):
        """Validate that folder and filename are provided."""
        if not self.folder_edit.text().strip():
            QMessageBox.warning(
                self,
                "No Folder Selected",
                "Please select a save folder."
            )
            return False
        
        if not self.base_filename_edit.text().strip():
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
    
    def _build_options_dict(self):
        """Build options dictionary for save operations."""
        return {
            'folder': self.folder_edit.text().strip(),
            'base_filename': self.base_filename_edit.text().strip(),
            'margin_percent': self.margin_spin.value(),
            'delimiter': self._get_delimiter(),
            'delimiter_name': self.delimiter_combo.currentText().lower(),
        }
    
    def _save_image_margin(self):
        """Save image with margin around data."""
        if not self._validate_inputs():
            return
        
        options = self._build_options_dict()
        options['filename'] = self.image_margin_filename.text().strip()
        options['use_view_range'] = False
        
        self.status_label.setText("Saving image (margin)...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
        self.save_image_with_margin_requested.emit(options)
    
    def _save_image_view(self):
        """Save image using current plot view range."""
        if not self._validate_inputs():
            return
        
        options = self._build_options_dict()
        options['filename'] = self.image_view_filename.text().strip()
        options['use_view_range'] = True
        
        self.status_label.setText("Saving image (view)...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
        self.save_image_view_range_requested.emit(options)
    
    def _save_ascii(self):
        """Save ASCII data."""
        if not self._validate_inputs():
            return
        
        options = self._build_options_dict()
        options['filename'] = self.ascii_filename.text().strip()
        
        self.status_label.setText("Saving ASCII data...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
        self.save_ascii_requested.emit(options)
    
    def _save_params(self):
        """Save parameters."""
        if not self._validate_inputs():
            return
        
        options = self._build_options_dict()
        options['filename'] = self.params_filename.text().strip()
        
        self.status_label.setText("Saving parameters...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
        self.save_parameters_requested.emit(options)
    
    def _save_all(self):
        """Save all formats."""
        if not self._validate_inputs():
            return
        
        options = self._build_options_dict()
        options['image_margin_filename'] = self.image_margin_filename.text().strip()
        options['image_view_filename'] = self.image_view_filename.text().strip()
        options['ascii_filename'] = self.ascii_filename.text().strip()
        options['params_filename'] = self.params_filename.text().strip()
        
        self.status_label.setText("Saving all formats...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 9pt; }")
        self.save_all_requested.emit(options)
    
    def set_status(self, message, is_success=True):
        """Set status message with appropriate styling."""
        self.status_label.setText(message)
        if is_success:
            self.status_label.setStyleSheet("QLabel { color: green; font-size: 9pt; }")
        else:
            self.status_label.setStyleSheet("QLabel { color: red; font-size: 9pt; }")
    
    def update_base_filename(self, filename):
        """Update the base filename (called when new file is loaded)."""
        self.base_filename_edit.setText(filename)
        self._update_preview_filenames()
    
    def update_folder(self, folder):
        """Update the save folder."""
        self.folder_edit.setText(folder)
    
    def set_delimiter_from_config(self, delimiter_name):
        """Set the delimiter combo from config value."""
        if delimiter_name == "comma":
            self.delimiter_combo.setCurrentIndex(0)
        elif delimiter_name == "tab":
            self.delimiter_combo.setCurrentIndex(1)
        elif delimiter_name == "space":
            self.delimiter_combo.setCurrentIndex(2)
    
    def get_delimiter_name(self):
        """Get the delimiter name for config storage."""
        return self.delimiter_combo.currentText().lower()
