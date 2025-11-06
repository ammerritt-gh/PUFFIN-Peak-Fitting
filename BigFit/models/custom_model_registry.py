"""
Custom Model Registry - Manages persistence and CRUD for user-defined composite models.

Custom models are stored as JSON files in config/custom_models/ with the following structure:
{
  "name": "My Custom Model",
  "components": [
    {
      "base_spec": "gaussian",
      "label": "Gaussian 1",
      "params": {"Area": 1.0, "Width": 1.0, "Center": 0.0}
    },
    {
      "base_spec": "lorentzian",
      "label": "Lorentzian 1",
      "params": {"Area": 0.5, "Width": 0.5, "Center": 5.0}
    }
  ]
}
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class CustomModelRegistry:
    """Singleton registry for managing custom composite models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Determine config directory path
        # The config folder is at BigFit/config/custom_models
        try:
            # Get the models directory (where this file is located)
            models_dir = Path(__file__).parent
            # Go up to BigFit directory, then down to config/custom_models
            bigfit_dir = models_dir.parent
            self.custom_models_dir = bigfit_dir / "config" / "custom_models"
        except Exception:
            # Fallback to relative path
            self.custom_models_dir = Path("config/custom_models")
        
        # Ensure directory exists
        self.custom_models_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of custom models: {name: model_data}
        self._models: Dict[str, Dict[str, Any]] = {}
        
        # Load all models from disk
        self._load_all()
    
    def _load_all(self):
        """Load all custom model JSON files from disk."""
        self._models.clear()
        
        if not self.custom_models_dir.exists():
            return
        
        for file_path in self.custom_models_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    name = data.get("name")
                    if name:
                        self._models[name] = data
            except Exception as e:
                print(f"Failed to load custom model from {file_path}: {e}")
    
    def _save_model(self, name: str):
        """Save a single model to disk."""
        if name not in self._models:
            return
        
        # Create safe filename from model name
        safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
        safe_name = safe_name.strip().replace(' ', '_')
        file_path = self.custom_models_dir / f"{safe_name}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._models[name], f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save custom model '{name}': {e}")
    
    def _delete_model_file(self, name: str):
        """Delete the JSON file for a model."""
        safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
        safe_name = safe_name.strip().replace(' ', '_')
        file_path = self.custom_models_dir / f"{safe_name}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Failed to delete custom model file '{file_path}': {e}")
    
    # Public API
    
    def get_custom_model_names(self) -> List[str]:
        """Return list of all custom model names."""
        return list(self._models.keys())
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a custom model by name. Returns None if not found."""
        return self._models.get(name)
    
    def create_model(self, name: str) -> bool:
        """Create a new empty custom model. Returns True on success."""
        if not name or not name.strip():
            return False
        
        name = name.strip()
        
        # Check if model already exists
        if name in self._models:
            return False
        
        # Create new model structure
        self._models[name] = {
            "name": name,
            "components": []
        }
        
        # Save to disk
        try:
            self._save_model(name)
            return True
        except Exception:
            # Rollback in-memory state
            del self._models[name]
            return False
    
    def delete_model(self, name: str) -> bool:
        """Delete a custom model. Returns True on success."""
        if name not in self._models:
            return False
        
        # Remove from memory
        del self._models[name]
        
        # Remove file
        self._delete_model_file(name)
        
        return True
    
    def add_component(self, model_name: str, base_spec: str, label: str, params: Dict[str, Any]) -> bool:
        """Add a component to a custom model. Returns True on success."""
        if model_name not in self._models:
            return False
        
        component = {
            "base_spec": base_spec,
            "label": label,
            "params": params
        }
        
        self._models[model_name]["components"].append(component)
        
        try:
            self._save_model(model_name)
            return True
        except Exception:
            # Rollback
            self._models[model_name]["components"].pop()
            return False
    
    def remove_component(self, model_name: str, index: int) -> bool:
        """Remove a component from a custom model by index. Returns True on success."""
        if model_name not in self._models:
            return False
        
        components = self._models[model_name]["components"]
        if index < 0 or index >= len(components):
            return False
        
        # Backup for rollback
        removed = components.pop(index)
        
        try:
            self._save_model(model_name)
            return True
        except Exception:
            # Rollback
            components.insert(index, removed)
            return False
    
    def move_component(self, model_name: str, old_index: int, new_index: int) -> bool:
        """Move a component within a custom model. Returns True on success."""
        if model_name not in self._models:
            return False
        
        components = self._models[model_name]["components"]
        if old_index < 0 or old_index >= len(components):
            return False
        if new_index < 0 or new_index >= len(components):
            return False
        
        # Perform the move
        component = components.pop(old_index)
        components.insert(new_index, component)
        
        try:
            self._save_model(model_name)
            return True
        except Exception:
            # Rollback
            components.pop(new_index)
            components.insert(old_index, component)
            return False
    
    def update_component_params(self, model_name: str, index: int, params: Dict[str, Any]) -> bool:
        """Update the parameters of a component. Returns True on success."""
        if model_name not in self._models:
            return False
        
        components = self._models[model_name]["components"]
        if index < 0 or index >= len(components):
            return False
        
        # Backup for rollback
        old_params = components[index]["params"].copy()
        
        # Update params
        components[index]["params"] = params
        
        try:
            self._save_model(model_name)
            return True
        except Exception:
            # Rollback
            components[index]["params"] = old_params
            return False
    
    def reload(self):
        """Reload all models from disk."""
        self._load_all()


# Singleton accessor
def get_custom_model_registry() -> CustomModelRegistry:
    """Get the singleton CustomModelRegistry instance."""
    return CustomModelRegistry()
