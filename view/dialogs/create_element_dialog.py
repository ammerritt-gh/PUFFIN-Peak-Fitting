"""
Dialog for creating a new model element via GUI.

This dialog allows users to:
- Define a new model element with parameters
- Write Python evaluation code
- Test the code with default parameters
- Save as a new element YAML file
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QFormLayout, QGroupBox,
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class CreateElementDialog(QDialog):
    """Dialog for creating a new model element."""

    def __init__(self, parent=None):
        """
        Initialize the create element dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.save_path = None
        
        self.setWindowTitle("Create New Model Element")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Element info section
        info_group = QGroupBox("Element Information")
        info_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., My Custom Peak")
        info_layout.addRow("Element Name:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Description of this model element")
        self.description_edit.setMaximumHeight(60)
        info_layout.addRow("Description:", self.description_edit)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["peak", "background", "other"])
        info_layout.addRow("Category:", self.category_combo)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Parameters section
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        
        param_layout.addWidget(QLabel("Define parameters for this element:"))
        
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(5)
        self.param_table.setHorizontalHeaderLabels(["Name", "Default Value", "Min", "Max", "Description"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.param_table.setRowCount(3)  # Start with 3 rows
        
        # Add some default rows
        for i in range(3):
            self.param_table.setItem(i, 0, QTableWidgetItem(""))
            self.param_table.setItem(i, 1, QTableWidgetItem("1.0"))
            self.param_table.setItem(i, 2, QTableWidgetItem(""))
            self.param_table.setItem(i, 3, QTableWidgetItem(""))
            self.param_table.setItem(i, 4, QTableWidgetItem(""))
        
        param_layout.addWidget(self.param_table)
        
        # Add/Remove row buttons
        param_btn_row = QHBoxLayout()
        add_row_btn = QPushButton("Add Parameter")
        add_row_btn.clicked.connect(self._add_parameter_row)
        remove_row_btn = QPushButton("Remove Parameter")
        remove_row_btn.clicked.connect(self._remove_parameter_row)
        param_btn_row.addWidget(add_row_btn)
        param_btn_row.addWidget(remove_row_btn)
        param_btn_row.addStretch()
        param_layout.addLayout(param_btn_row)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Evaluation code section
        code_group = QGroupBox("Evaluation Code")
        code_layout = QVBoxLayout()
        
        instructions = QLabel(
            "Enter the expression to calculate y values from x.\n"
            "Start with: y = \n"
            "Available: x (input array), parameter names (defined above)\n"
            "Functions: np, sqrt, log, exp, pi, sin, cos, tan\n"
            "\n"
            "Examples:\n"
            "  y = Amplitude * np.exp(-((x - Center) / Width)**2)  # Gaussian\n"
            "  y = Amplitude / (1 + ((x - Center) / Width)**2)      # Lorentzian\n"
            "  y = Slope * x + Intercept                            # Linear"
        )
        instructions.setWordWrap(True)
        code_layout.addWidget(instructions)
        
        self.code_edit = QTextEdit()
        self.code_edit.setPlainText("y = ")
        self.code_edit.setFontFamily("Courier")
        code_layout.addWidget(self.code_edit)
        
        # Test button
        test_btn = QPushButton("Test Code")
        test_btn.clicked.connect(self._test_code)
        code_layout.addWidget(test_btn)
        
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setMaximumHeight(100)
        self.test_output.setPlaceholderText("Test output will appear here...")
        code_layout.addWidget(self.test_output)
        
        code_group.setLayout(code_layout)
        layout.addWidget(code_group)
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton("Save Element")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)

    def _add_parameter_row(self):
        """Add a new parameter row to the table."""
        row = self.param_table.rowCount()
        self.param_table.insertRow(row)
        self.param_table.setItem(row, 0, QTableWidgetItem(""))
        self.param_table.setItem(row, 1, QTableWidgetItem("1.0"))
        self.param_table.setItem(row, 2, QTableWidgetItem(""))
        self.param_table.setItem(row, 3, QTableWidgetItem(""))
        self.param_table.setItem(row, 4, QTableWidgetItem(""))

    def _remove_parameter_row(self):
        """Remove the selected parameter row."""
        current_row = self.param_table.currentRow()
        if current_row >= 0:
            self.param_table.removeRow(current_row)

    def _get_parameters(self) -> List[Dict[str, Any]]:
        """Extract parameters from the table."""
        parameters = []
        for row in range(self.param_table.rowCount()):
            name_item = self.param_table.item(row, 0)
            if not name_item or not name_item.text().strip():
                continue
            
            name = name_item.text().strip()
            value_item = self.param_table.item(row, 1)
            min_item = self.param_table.item(row, 2)
            max_item = self.param_table.item(row, 3)
            desc_item = self.param_table.item(row, 4)
            
            param = {
                'name': name,
                'description': desc_item.text().strip() if desc_item else "",
                'type': 'float',
                'value': float(value_item.text()) if value_item and value_item.text().strip() else 1.0,
                'decimals': 3,
                'step': 0.1,
            }
            
            if min_item and min_item.text().strip():
                try:
                    param['min'] = float(min_item.text())
                except ValueError:
                    pass
            
            if max_item and max_item.text().strip():
                try:
                    param['max'] = float(max_item.text())
                except ValueError:
                    pass
            
            parameters.append(param)
        
        return parameters

    def _test_code(self):
        """Test the evaluation code with default parameters."""
        self.test_output.clear()
        
        try:
            # Get parameters
            parameters = self._get_parameters()
            if not parameters:
                self.test_output.setPlainText("❌ Error: No parameters defined")
                return
            
            # Get code
            code = self.code_edit.toPlainText().strip()
            if not code:
                self.test_output.setPlainText("❌ Error: No evaluation code provided")
                return
            
            # Check if code starts with "y = " and convert it to "return "
            if code.startswith("y = "):
                code = "return " + code[4:]
            elif code.startswith("y="):
                code = "return " + code[2:]
            elif not code.startswith("return "):
                # If it doesn't have "return" or "y =", assume it's the expression and add "return"
                code = "return " + code
            
            # Create test environment
            import numpy as np
            from numpy import sqrt, log, exp, pi, sin, cos, tan
            
            # Create parameter dict with default values
            param_dict = {p['name']: p['value'] for p in parameters}
            
            # Create test x array
            x = np.linspace(-10, 10, 21)
            
            # Build evaluation function
            param_names = [p['name'] for p in parameters]
            param_args = ', '.join(param_names)
            
            func_code = f"def evaluate(x, {param_args}):\n"
            for line in code.split('\n'):
                func_code += "    " + line + "\n"
            
            # Test compilation
            local_env = {'np': np, 'sqrt': sqrt, 'log': log, 'exp': exp, 'pi': pi,
                        'sin': sin, 'cos': cos, 'tan': tan}
            exec(func_code, local_env)
            evaluate_func = local_env['evaluate']
            
            # Test execution
            result = evaluate_func(x, **param_dict)
            
            # Validate result
            if not isinstance(result, np.ndarray):
                result = np.asarray(result, dtype=float)
            
            # Ensure result is the same shape as x by flattening if needed
            if result.ndim > 1:
                result = result.flatten()
            
            if len(result) != len(x):
                self.test_output.setPlainText(
                    f"❌ Error: Result length {len(result)} doesn't match input length {len(x)}\n"
                    f"Make sure your expression returns one value per input x value."
                )
            else:
                self.test_output.setPlainText(
                    f"✅ Success! Code compiled and executed.\n"
                    f"Input: {len(x)} points from {x[0]:.2f} to {x[-1]:.2f}\n"
                    f"Output: min={result.min():.4g}, max={result.max():.4g}, mean={result.mean():.4g}"
                )
            
        except SyntaxError as e:
            self.test_output.setPlainText(f"❌ Syntax Error:\n{e}")
        except Exception as e:
            self.test_output.setPlainText(f"❌ Error:\n{e}")

    def _validate_inputs(self) -> bool:
        """Validate user inputs before saving."""
        name = self.name_edit.text().strip()
        
        if not name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Please enter an element name."
            )
            return False
        
        # Check for valid name characters
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\s-]*$', name):
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Element name must start with a letter and contain only letters, numbers, spaces, hyphens, and underscores."
            )
            return False
        
        # Check parameters
        parameters = self._get_parameters()
        if not parameters:
            QMessageBox.warning(
                self,
                "No Parameters",
                "Please define at least one parameter."
            )
            return False
        
        # Check code
        code = self.code_edit.toPlainText().strip()
        if not code:
            QMessageBox.warning(
                self,
                "No Code",
                "Please enter evaluation code."
            )
            return False
        
        return True

    def _on_save(self):
        """Handle save button click."""
        if not self._validate_inputs():
            return
        
        name = self.name_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        category = self.category_combo.currentText()
        parameters = self._get_parameters()
        code = self.code_edit.toPlainText().strip()
        
        # Convert "y = " syntax to just the expression
        if code.startswith("y = "):
            code = code[4:]
        elif code.startswith("y="):
            code = code[2:]
        
        # Generate filename
        filename = name.lower().replace(' ', '_').replace('-', '_')
        filename = re.sub(r'[^a-z0-9_]', '', filename)
        
        if not filename:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Could not generate valid filename from element name."
            )
            return
        
        # Get save path
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_dir = repo_root / "models" / "model_elements"
            default_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            default_dir = Path.home()
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Element",
            str(default_dir / f"{filename}.yaml"),
            "YAML Files (*.yaml)"
        )
        
        if not filepath:
            return
        
        self.save_path = Path(filepath)
        
        # Build YAML structure
        import yaml
        
        yaml_data = {
            'name': name,
            'description': description or f"Custom element: {name}",
            'version': 1,
            'author': 'BigFit User',
            'category': category,
            'parameters': parameters,
            'evaluate': code  # Save as single-line string
        }
        
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False,
                         allow_unicode=True, indent=2)
            
            # Try to reload elements immediately
            reload_success = False
            try:
                from models import reload_model_elements
                reload_model_elements()
                reload_success = True
            except Exception as reload_err:
                pass
            
            if reload_success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Element saved to:\n{self.save_path}\n\n"
                    "Element loaded! It's now available for use in Custom Model."
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Element saved to:\n{self.save_path}\n\n"
                    "Restart BigFit to use the new element."
                )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save element:\n{e}"
            )

    def get_save_path(self) -> Optional[Path]:
        """Get the save path."""
        return self.save_path
