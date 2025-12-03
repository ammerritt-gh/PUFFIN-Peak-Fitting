"""
Example demonstrating input_handler usage patterns.

This example shows how to use the integrated input_handler in the BigFit
application, with patterns similar to PySide_Fitter_PyQtGraph.py.
"""

# Example 1: Basic Setup (already done in main_window.py)
# =========================================================
def setup_input_handler_example():
    """Shows how InputHandler is integrated in MainWindow."""
    from view.input_handler import InputHandler
    
    # In MainWindow.__init__:
    # self.input_handler = InputHandler(self.plot_widget)
    # self._connect_input_handler()
    
    # In MainWindow._connect_input_handler():
    # self.input_handler.mouse_clicked.connect(self._on_plot_clicked)
    # self.input_handler.mouse_moved.connect(self._on_plot_mouse_moved)
    # self.input_handler.key_pressed.connect(self._on_plot_key_pressed)
    # self.input_handler.wheel_scrolled.connect(self._on_plot_wheel_scrolled)
    pass


# Example 2: Handling Click Events
# =================================
def handle_plot_click_example(x, y, button):
    """
    Example handler for plot click events.
    This would be in MainWindow._on_plot_clicked().
    """
    from PySide6.QtCore import Qt
    
    # Check which button was clicked
    try:
        if button == Qt.LeftButton:
            print(f"Left click at ({x:.2f}, {y:.2f})")
            # Find nearest peak and select it
            # self.viewmodel.select_nearest_peak(x, y)
        elif button == Qt.RightButton:
            print(f"Right click at ({x:.2f}, {y:.2f})")
            # Show context menu
            # self.show_context_menu(x, y)
    except:
        pass
    
    # Always delegate to viewmodel
    # self.viewmodel.handle_plot_click(x, y, button)


# Example 3: Implementing Drag Operations
# ========================================
class DragExample:
    """Example showing how to implement dragging."""
    
    def __init__(self):
        self._dragging = False
        self._drag_start_x = None
        self._drag_peak = None
    
    def _on_plot_clicked(self, x, y, button):
        """Start drag on left click."""
        from PySide6.QtCore import Qt
        
        if button == Qt.LeftButton:
            # Find peak near click position
            peak = self._find_peak_at(x, y)
            if peak:
                self._dragging = True
                self._drag_start_x = x
                self._drag_peak = peak
    
    def _on_plot_mouse_moved(self, x, y):
        """Update peak position during drag."""
        if self._dragging and self._drag_peak:
            # Update peak center
            delta = x - self._drag_start_x
            # self.viewmodel.move_peak(self._drag_peak, delta)
            self._drag_start_x = x
    
    def _on_mouse_released(self):
        """End drag operation."""
        self._dragging = False
        self._drag_start_x = None
        self._drag_peak = None
    
    def _find_peak_at(self, x, y):
        """Find peak near coordinates (placeholder)."""
        # Would search through peaks and return nearest
        return None


# Example 4: Keyboard Shortcut Handling
# ======================================
def handle_key_press_example(key, modifiers):
    """
    Example keyboard handler with various shortcuts.
    This would be in FitterViewModel.handle_key_press().
    """
    from PySide6.QtCore import Qt
    
    # Check modifiers
    is_ctrl = bool(modifiers & Qt.ControlModifier)
    is_shift = bool(modifiers & Qt.ShiftModifier)
    is_alt = bool(modifiers & Qt.AltModifier)
    
    # Handle shortcuts
    if key == Qt.Key_F and not is_ctrl:
        # F - Run fit
        print("Running fit...")
        # self.run_fit()
    
    elif key == Qt.Key_F and is_ctrl:
        # Ctrl+F - Find/search
        print("Opening find dialog...")
        # self.show_find_dialog()
    
    elif key == Qt.Key_S and is_ctrl:
        # Ctrl+S - Save
        print("Saving data...")
        # self.save_data()
    
    elif key == Qt.Key_Z and is_ctrl:
        # Ctrl+Z - Undo
        if is_shift:
            # Ctrl+Shift+Z - Redo
            print("Redo last action")
            # self.redo()
        else:
            print("Undo last action")
            # self.undo()
    
    elif key == Qt.Key_Delete:
        # Delete - Remove selected peak
        print("Deleting selected peak...")
        # self.delete_selected_peak()
    
    elif Qt.Key_1 <= key <= Qt.Key_9:
        # Number keys 1-9 - Select peak by index
        index = key - Qt.Key_1
        print(f"Selecting peak {index + 1}")
        # self.select_peak_by_index(index)


# Example 5: Parameter Adjustment with Mouse Wheel
# =================================================
def handle_wheel_scroll_example(delta, modifiers):
    """
    Example wheel handler for parameter adjustment.
    This would be in FitterViewModel.handle_wheel_scroll().
    """
    from PySide6.QtCore import Qt
    
    # Determine scroll direction
    step = 1 if delta > 0 else -1
    
    # Check modifiers
    is_ctrl = bool(modifiers & Qt.ControlModifier)
    is_shift = bool(modifiers & Qt.ShiftModifier)
    is_alt = bool(modifiers & Qt.AltModifier)
    
    # Different modifier combinations adjust different parameters
    if is_ctrl and not is_shift and not is_alt:
        # Ctrl+Wheel - Adjust gaussian width
        print(f"Adjusting gaussian width, step={step}")
        # current = self.get_parameter('gauss_fwhm')
        # new_value = current * (1.1 ** step)
        # self.apply_parameters({'gauss_fwhm': new_value})
    
    elif is_ctrl and is_shift:
        # Ctrl+Shift+Wheel - Adjust lorentzian width
        print(f"Adjusting lorentzian width, step={step}")
        # Similar to above
    
    elif is_shift and not is_ctrl:
        # Shift+Wheel - Adjust damping (for selected peak)
        print(f"Adjusting damping for selected peak, step={step}")
        # if self.selected_peak:
        #     current = self.selected_peak.damping
        #     new_damping = current * (1.05 ** step)
        #     self.update_peak_damping(self.selected_peak, new_damping)
    
    elif not is_ctrl and not is_shift and not is_alt:
        # No modifiers + Wheel - Adjust height (for selected peak)
        print(f"Adjusting height for selected peak, step={step}")
        # Similar to damping adjustment


# Example 6: Peak Selection Logic
# ================================
class PeakSelectionExample:
    """Example showing peak selection implementation."""
    
    def __init__(self):
        self.selected_peak = None
        self.peaks = []  # List of peak objects
    
    def handle_plot_click(self, x, y, button):
        """Select peak on click."""
        from PySide6.QtCore import Qt
        
        if button == Qt.LeftButton:
            # Find nearest peak
            nearest = self._find_nearest_peak(x, y, threshold=50)  # 50 pixels
            
            if nearest:
                # Select the peak
                self.select_peak(nearest)
            else:
                # Click was not near any peak - clear selection
                self.clear_selection()
    
    def _find_nearest_peak(self, x, y, threshold=50):
        """
        Find peak nearest to coordinates.
        
        Args:
            x: X coordinate in data space
            y: Y coordinate in data space
            threshold: Maximum pixel distance to consider
            
        Returns:
            Peak object if found within threshold, None otherwise
        """
        # Would convert threshold to data coordinates
        # and search through self.peaks
        
        # Example logic:
        # best_peak = None
        # min_distance = float('inf')
        # 
        # for peak in self.peaks:
        #     dx = abs(peak.center - x)
        #     dy = abs(peak.height - y)
        #     distance = sqrt(dx**2 + dy**2)
        #     
        #     if distance < min_distance and distance < threshold:
        #         min_distance = distance
        #         best_peak = peak
        # 
        # return best_peak
        
        return None
    
    def select_peak(self, peak):
        """Select a peak and update UI."""
        self.selected_peak = peak
        print(f"Selected peak at {peak.center if hasattr(peak, 'center') else '?'}")
        # Update visual highlighting
        # self.update_plot()
    
    def clear_selection(self):
        """Clear peak selection."""
        self.selected_peak = None
        print("Selection cleared")
        # Update visual highlighting
        # self.update_plot()


# Example 7: Box Selection for Excluding Points
# ==============================================
class BoxSelectionExample:
    """Example showing box selection for excluding data points."""
    
    def __init__(self):
        self._box_start = None
        self._box_active = False
        self.exclude_mode = False
        self.excluded_indices = set()
    
    def toggle_exclude_mode(self):
        """Toggle exclude mode on/off."""
        self.exclude_mode = not self.exclude_mode
        print(f"Exclude mode: {'ON' if self.exclude_mode else 'OFF'}")
    
    def _on_plot_clicked(self, x, y, button):
        """Start box selection in exclude mode."""
        from PySide6.QtCore import Qt
        
        if self.exclude_mode and button == Qt.LeftButton:
            # Start box selection
            self._box_start = (x, y)
            self._box_active = True
    
    def _on_plot_mouse_moved(self, x, y):
        """Update box during drag."""
        if self._box_active and self._box_start:
            # Draw selection box from _box_start to (x, y)
            # self.draw_selection_box(self._box_start, (x, y))
            pass
    
    def _on_mouse_released(self, x, y):
        """Finish box selection."""
        if self._box_active and self._box_start:
            # Find points within box
            x1, y1 = self._box_start
            x2, y2 = x, y
            
            # Toggle exclusion for points in box
            # points_in_box = self._find_points_in_box(x1, y1, x2, y2)
            # for idx in points_in_box:
            #     if idx in self.excluded_indices:
            #         self.excluded_indices.remove(idx)
            #     else:
            #         self.excluded_indices.add(idx)
            
            # Update plot
            # self.update_plot()
            
            # Reset box state
            self._box_active = False
            self._box_start = None


# Example 8: Integration in main.py
# ==================================
def main_integration_example():
    """
    Shows how all pieces connect in main.py.
    """
    # This is the existing pattern in main.py, now enhanced with input handler
    
    # from PySide6.QtWidgets import QApplication
    # from models import ModelState
    # from view.main_window import MainWindow
    # from viewmodel.fitter_vm import FitterViewModel
    # 
    # app = QApplication(sys.argv)
    # 
    # # Create MVVM components
    # model_state = ModelState()
    # viewmodel = FitterViewModel(model_state)
    # window = MainWindow(viewmodel)
    # 
    # # Connect signals (ViewModel → View)
    # viewmodel.plot_updated.connect(window.update_plot_data)
    # viewmodel.log_message.connect(window.append_log)
    # 
    # # Input handler is automatically created and connected in MainWindow.__init__
    # # See MainWindow._connect_input_handler() for signal connections
    # 
    # window.show()
    # sys.exit(app.exec())
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("Input Handler Usage Examples")
    print("=" * 60)
    print("\nThis file contains examples of how to use the input_handler")
    print("integration. See the function docstrings for details.")
    print("\nAvailable examples:")
    print("  1. setup_input_handler_example - Basic setup")
    print("  2. handle_plot_click_example - Click handling")
    print("  3. DragExample - Drag operations")
    print("  4. handle_key_press_example - Keyboard shortcuts")
    print("  5. handle_wheel_scroll_example - Wheel parameter adjustment")
    print("  6. PeakSelectionExample - Peak selection")
    print("  7. BoxSelectionExample - Box selection for exclusions")
    print("  8. main_integration_example - Full integration")
    print("\nSee INPUT_HANDLER_INTEGRATION.md for comprehensive documentation.")
