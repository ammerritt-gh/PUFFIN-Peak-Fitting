"""
Dialog for saving a custom model to a YAML file.

This dialog allows users to:
- View all components/elements of the current model
- See parameter values, fixed state, link groups, and bounds
- Specify a name for the model
- Choose the save location (default: models/model_elements/)
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTextEdit
)
from PySide6.QtCore import Qt
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class SaveModelDialog(QDialog):
    """Dialog for saving a custom composite model to YAML."""

    def __init__(self, model_data: Dict[str, Any], parent=None):
        """
        Initialize the save model dialog.

        Args:
            model_data: Dictionary containing model structure and parameters
                Expected keys: 'name', 'components', 'default_save_path'
            parent: Parent widget
        """
        super().__init__(parent)
        self.model_data = model_data
        self.save_path = None
        
        self.setWindowTitle("Save Custom Model")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._init_ui()
        self._populate_tree()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Model name section
        name_group = QGroupBox("Model Information")
        name_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        suggested_name = self.model_data.get('name', 'CustomModel')
        self.name_edit.setText(suggested_name)
        self.name_edit.setPlaceholderText("Enter model name (e.g., MyCustomModel)")
        name_layout.addRow("Model Name:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Optional: Enter a description for this model")
        self.description_edit.setMaximumHeight(80)
        name_layout.addRow("Description:", self.description_edit)
        
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # Model structure section
        structure_group = QGroupBox("Model Structure")
        structure_layout = QVBoxLayout()
        
        structure_layout.addWidget(QLabel("Components and Parameters:"))
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels([
            "Element/Parameter", "Value", "Fixed", "Link Group", "Min", "Max"
        ])
        self.tree_widget.setAlternatingRowColors(True)
        # Allow column resizing
        for i in range(6):
            self.tree_widget.resizeColumnToContents(i)
        
        structure_layout.addWidget(self.tree_widget)
        structure_group.setLayout(structure_layout)
        layout.addWidget(structure_group)
        
        # Save location section
        loc_group = QGroupBox("Save Location")
        loc_layout = QVBoxLayout()
        
        loc_row = QHBoxLayout()
        self.location_edit = QLineEdit()
        default_path = self.model_data.get('default_save_path', '')
        self.location_edit.setText(str(default_path))
        self.location_edit.setReadOnly(True)
        loc_row.addWidget(self.location_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_location)
        loc_row.addWidget(browse_btn)
        
        loc_layout.addLayout(loc_row)
        loc_layout.addWidget(QLabel("The model will be saved as a .yaml file"))
        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)

    def _populate_tree(self):
        """Populate the tree widget with model components and parameters."""
        self.tree_widget.clear()
        
        components = self.model_data.get('components', [])
        
        if not components:
            item = QTreeWidgetItem(self.tree_widget)
            item.setText(0, "No components to display")
            return
        
        for comp_idx, component in enumerate(components):
            # Create component item
            comp_type = component.get('type', 'Unknown')
            comp_prefix = component.get('prefix', '')
            comp_label = component.get('label', comp_type)
            
            comp_item = QTreeWidgetItem(self.tree_widget)
            comp_item.setText(0, f"Component {comp_idx + 1}: {comp_label}")
            comp_item.setExpanded(True)
            
            # Set component color if available
            color = component.get('color')
            if color:
                try:
                    from PySide6.QtGui import QBrush, QColor
                    comp_item.setForeground(0, QBrush(QColor(color)))
                except Exception:
                    pass
            
            # Add parameters as children
            parameters = component.get('parameters', {})
            for param_name, param_data in parameters.items():
                param_item = QTreeWidgetItem(comp_item)
                param_item.setText(0, f"  {param_name}")
                
                # Value
                value = param_data.get('value', '')
                if isinstance(value, float):
                    param_item.setText(1, f"{value:.6g}")
                else:
                    param_item.setText(1, str(value))
                
                # Fixed state
                fixed = param_data.get('fixed', False)
                param_item.setText(2, "Yes" if fixed else "No")
                
                # Link group
                link_group = param_data.get('link_group')
                if link_group is not None and link_group != 0:
                    param_item.setText(3, str(link_group))
                else:
                    param_item.setText(3, "")
                
                # Min/Max bounds
                min_val = param_data.get('min')
                if min_val is not None:
                    param_item.setText(4, f"{min_val:.6g}")
                else:
                    param_item.setText(4, "")
                
                max_val = param_data.get('max')
                if max_val is not None:
                    param_item.setText(5, f"{max_val:.6g}")
                else:
                    param_item.setText(5, "")
        
        # Resize columns to fit content
        for i in range(6):
            self.tree_widget.resizeColumnToContents(i)

    def _browse_location(self):
        """Open a file dialog to choose save location."""
        current_path = self.location_edit.text()
        if not current_path:
            current_path = str(Path.home())
        
        # Get directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.location_edit.setText(directory)

    def _validate_inputs(self) -> bool:
        """Validate user inputs before saving."""
        name = self.name_edit.text().strip()
        
        if not name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Please enter a model name."
            )
            return False
        
        # Check for valid filename characters
        # Allow alphanumeric, spaces, hyphens, underscores
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\s-]*$', name):
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Model name must start with a letter and contain only letters, numbers, spaces, hyphens, and underscores."
            )
            return False
        
        # Check that sanitization won't result in empty filename
        test_filename = name.lower().replace(' ', '_').replace('-', '_')
        test_filename = re.sub(r'[^a-z0-9_]', '', test_filename)
        if not test_filename:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Model name must contain at least one alphanumeric character after conversion to filename."
            )
            return False
        
        location = self.location_edit.text().strip()
        if not location:
            QMessageBox.warning(
                self,
                "Invalid Location",
                "Please select a save location."
            )
            return False
        
        # Check if location exists
        if not Path(location).exists():
            reply = QMessageBox.question(
                self,
                "Create Directory",
                f"The directory '{location}' does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return False
        
        return True

    def _on_save(self):
        """Handle save button click."""
        if not self._validate_inputs():
            return
        
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        
        # Generate filename from model name
        # Replace spaces with underscores, make lowercase
        filename = name.lower().replace(' ', '_').replace('-', '_')
        filename = re.sub(r'[^a-z0-9_]', '', filename)  # Remove invalid chars
        
        # Ensure filename is not empty after sanitization
        if not filename:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Model name must contain at least one alphanumeric character."
            )
            return
        
        self.save_path = Path(location) / f"{filename}.yaml"
        
        # Check if file exists
        if self.save_path.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite File",
                f"The file '{self.save_path.name}' already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.accept()

    def get_model_name(self) -> str:
        """Get the entered model name."""
        return self.name_edit.text().strip()

    def get_description(self) -> str:
        """Get the entered description."""
        return self.description_edit.toPlainText().strip()

    def get_save_path(self) -> Optional[Path]:
        """Get the save path."""
        return self.save_path
