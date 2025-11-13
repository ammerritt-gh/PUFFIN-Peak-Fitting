"""
Elements dock widget for managing model elements/components.
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QListWidget, QAbstractItemView, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush


class ElementsDock(QDockWidget):
    """Dock widget for managing model elements (peaks/components)."""

    # Signals
    element_selected = Signal(int)  # row number
    element_add_clicked = Signal()
    element_remove_clicked = Signal()
    element_rows_moved = Signal(object, int, int, object, int)  # parent, start, end, dest_parent, dest_row

    def __init__(self, parent=None):
        """
        Initialize the elements dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Elements", parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # light-weight element list
        self.element_list = QListWidget()
        self.element_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.element_list.setDragEnabled(True)
        self.element_list.setAcceptDrops(True)
        self.element_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.element_list.setDefaultDropAction(Qt.MoveAction)
        vlayout.addWidget(self.element_list)

        # add / remove row
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Element")
        self.remove_btn = QPushButton("Remove Element")
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        vlayout.addLayout(btn_row)

        self.setWidget(container)

        # Connect internal signals
        self.element_list.currentRowChanged.connect(self._on_element_selected)
        self.element_list.model().rowsMoved.connect(self._on_element_rows_moved)
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)

    def _on_element_selected(self, row):
        """Handle element selection."""
        self.element_selected.emit(row)

    def _on_add_clicked(self):
        """Handle add button click."""
        self.element_add_clicked.emit()

    def _on_remove_clicked(self):
        """Handle remove button click."""
        self.element_remove_clicked.emit()

    def _on_element_rows_moved(self, parent, start, end, dest_parent, dest_row):
        """Handle drag-and-drop reordering."""
        self.element_rows_moved.emit(parent, start, end, dest_parent, dest_row)

    def refresh_element_list(self, descriptors):
        """
        Populate the element list from descriptors.

        Args:
            descriptors: List of element descriptor dicts with 'prefix', 'label', 'color', 'tooltip', etc.
        """
        try:
            self.element_list.blockSignals(True)
            self.element_list.clear()

            if descriptors:
                for desc in descriptors:
                    prefix = desc.get('prefix') or ''
                    label = desc.get('label') or prefix.rstrip('_') or 'element'
                    item = QListWidgetItem(label)
                    data = {'id': prefix, 'name': label}
                    if 'color' in desc:
                        data['color'] = desc['color']
                    item.setData(Qt.UserRole, data)
                    color = desc.get('color')
                    if color:
                        try:
                            item.setForeground(QBrush(QColor(color)))
                        except Exception:
                            pass
                    
                    # Handle special model entry
                    is_model = (prefix == 'model')
                    if is_model:
                        try:
                            f = item.font()
                            f.setBold(True)
                            item.setFont(f)
                        except Exception:
                            pass
                        try:
                            item.setBackground(QBrush(QColor("#f2f2f7")))
                            item.setToolTip("Active model (non-removable)")
                        except Exception:
                            pass
                        try:
                            item.setFlags(item.flags() & ~Qt.ItemIsDragEnabled)
                        except Exception:
                            pass
                    else:
                        try:
                            item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
                        except Exception:
                            pass
                    
                    # Add tooltip if provided
                    if 'tooltip' in desc:
                        try:
                            item.setToolTip(desc['tooltip'])
                        except Exception:
                            pass
                    
                    self.element_list.addItem(item)

            self.element_list.blockSignals(False)
        except Exception:
            try:
                self.element_list.blockSignals(False)
            except Exception:
                pass

    def get_element_count(self):
        """Get the number of elements in the list."""
        return self.element_list.count()

    def get_current_row(self):
        """Get the currently selected row."""
        return self.element_list.currentRow()

    def get_item(self, row):
        """Get the item at the specified row."""
        return self.element_list.item(row)

    def take_item(self, row):
        """Remove and return the item at the specified row."""
        return self.element_list.takeItem(row)

    def set_current_row(self, row):
        """Set the current row selection."""
        self.element_list.setCurrentRow(row)

    def clear_selection(self):
        """Clear the current selection."""
        self.element_list.clearSelection()

    def block_signals(self, block):
        """Block or unblock signals from the element list."""
        self.element_list.blockSignals(block)
