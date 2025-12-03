# dataio/fit_persistence.py
"""
Fit persistence module for saving and loading fit state.

Provides functionality to:
1. Save/load a generic 'default' fit that holds the last fit for any model data
2. Save/load file-specific fits tied to particular data files

The fits are stored as JSON files in the 'fits/' folder under the repo root.
"""
import json
import os
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Constants
DEFAULT_FIT_FILENAME = "default_fit.json"
FIT_FILE_VERSION = 1


def _get_fits_folder() -> Path:
    """Get the fits folder path (repo_root/fits/)."""
    # repo root is two levels up from this file: .../BigFit/dataio/fit_persistence.py
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "fits"


def _ensure_fits_folder() -> Path:
    """Ensure the fits folder exists and return its path."""
    fits_folder = _get_fits_folder()
    fits_folder.mkdir(parents=True, exist_ok=True)
    return fits_folder


def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Replace any non-alphanumeric characters (except . - _) with underscores
    sanitized = re.sub(r'[^\w\.\-]', '_', name)
    # Limit length to prevent filesystem issues
    return sanitized[:200] if len(sanitized) > 200 else sanitized


def _compute_file_signature(filepath: str) -> Dict[str, Any]:
    """Compute a signature for a data file to help match fits to files.

    Returns a dict with:
      - path: the original file path
      - size: file size in bytes
      - points: number of data points (if readable from fit context)
    """
    sig = {
        "path": filepath,
        "size": None,
        "points": None,
    }
    try:
        if filepath and os.path.isfile(filepath):
            sig["size"] = os.path.getsize(filepath)
    except Exception:
        pass
    return sig


def _get_fit_filename_for_file(filepath: str) -> str:
    """Generate a unique fit filename for a given data file path.

    Uses the basename of the file plus a short hash of the full path
    to handle cases where different folders have files with the same name.
    """
    basename = os.path.basename(filepath or "")
    if not basename:
        return "unknown_file_fit.json"

    # Create a short hash of the full path for uniqueness
    path_hash = hashlib.md5(filepath.encode('utf-8')).hexdigest()[:8]
    name_part = os.path.splitext(basename)[0]
    sanitized = _sanitize_filename(name_part)

    return f"{sanitized}_{path_hash}_fit.json"


def _extract_fit_state(model_state) -> Optional[Dict[str, Any]]:
    """Extract fit state from a ModelState object into a serializable dict.

    Args:
        model_state: The ModelState object to extract from

    Returns:
        A dict containing the fit state, or None if extraction fails
    """
    try:
        model_spec = getattr(model_state, "model_spec", None)
        if model_spec is None:
            return None

        params = {}
        fixed = {}
        link_groups = {}

        spec_params = getattr(model_spec, "params", {}) or {}
        for name, param in spec_params.items():
            try:
                params[name] = getattr(param, "value", None)
                fixed[name] = bool(getattr(param, "fixed", False))
                lg = getattr(param, "link_group", None)
                link_groups[name] = int(lg) if lg else None
            except Exception:
                params[name] = None
                fixed[name] = False
                link_groups[name] = None

        # Get fit_result if available
        fit_result = getattr(model_state, "fit_result", None)
        if fit_result is not None and not isinstance(fit_result, dict):
            fit_result = None

        # Get excluded mask
        excluded = []
        try:
            exc = getattr(model_state, "excluded", None)
            if exc is not None:
                import numpy as np
                excluded = np.asarray(exc, dtype=bool).tolist()
        except Exception:
            excluded = []

        # Get file info for signature
        file_info = getattr(model_state, "file_info", None) or {}
        filepath = file_info.get("path") if isinstance(file_info, dict) else None
        signature = _compute_file_signature(filepath) if filepath else {
            "path": None, "size": None, "points": None
        }

        # Add number of data points to signature
        try:
            x_data = getattr(model_state, "x_data", None)
            if x_data is not None:
                signature["points"] = len(x_data)
        except Exception:
            pass

        # Build the state dict
        state = {
            "version": FIT_FILE_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_name": getattr(model_state, "model_name", "Voigt"),
            "params": params,
            "fixed": fixed,
            "link_groups": link_groups,
            "fit_result": fit_result,
            "excluded": excluded,
            "signature": signature,
            "source": file_info if isinstance(file_info, dict) else {},
        }
        return state

    except Exception:
        return None


def _apply_fit_state(model_state, fit_data: Dict[str, Any], apply_excluded: bool = True) -> bool:
    """Apply saved fit state to a ModelState object.

    Args:
        model_state: The ModelState object to apply to
        fit_data: The fit state dict loaded from JSON
        apply_excluded: Whether to apply the excluded mask (may want to skip if data size differs)

    Returns:
        True if application succeeded, False otherwise
    """
    try:
        import numpy as np
        from models import get_model_spec

        # Get model spec, create if needed
        model_spec = getattr(model_state, "model_spec", None)
        saved_model_name = fit_data.get("model_name", "Voigt")

        # If model names differ or no spec, try to get the correct one
        current_model_name = getattr(model_state, "model_name", None)
        if model_spec is None or (current_model_name and current_model_name.lower() != saved_model_name.lower()):
            try:
                model_spec = get_model_spec(saved_model_name)
                setattr(model_state, "model_spec", model_spec)
                setattr(model_state, "model_name", saved_model_name)
            except Exception:
                pass

        if model_spec is None:
            return False

        # Apply parameters
        params = fit_data.get("params", {})
        fixed = fit_data.get("fixed", {})
        link_groups = fit_data.get("link_groups", {})

        spec_params = getattr(model_spec, "params", {}) or {}
        for name, value in params.items():
            if name in spec_params:
                try:
                    spec_params[name].value = value
                except Exception:
                    pass

        # Apply fixed states
        for name, is_fixed in fixed.items():
            if name in spec_params:
                try:
                    spec_params[name].fixed = bool(is_fixed)
                except Exception:
                    pass

        # Apply link groups
        for name, lg in link_groups.items():
            if name in spec_params:
                try:
                    spec_params[name].link_group = int(lg) if lg else None
                except Exception:
                    pass

        # Apply fit_result
        fit_result = fit_data.get("fit_result")
        if fit_result is not None:
            try:
                setattr(model_state, "fit_result", fit_result)
            except Exception:
                pass

        # Apply excluded mask if requested and data length matches
        if apply_excluded:
            excluded = fit_data.get("excluded", [])
            if excluded:
                try:
                    x_data = getattr(model_state, "x_data", None)
                    if x_data is not None and len(excluded) == len(x_data):
                        model_state.excluded = np.asarray(excluded, dtype=bool)
                except Exception:
                    pass

        return True

    except Exception:
        return False


def save_default_fit(model_state) -> bool:
    """Save the current fit as the generic default fit.

    This saves the fit state to default_fit.json, which can be loaded
    for any data as a starting point.

    Args:
        model_state: The ModelState object to save

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        fits_folder = _ensure_fits_folder()
        fit_data = _extract_fit_state(model_state)
        if fit_data is None:
            return False

        fit_path = fits_folder / DEFAULT_FIT_FILENAME
        tmp_path = fit_path.with_suffix(fit_path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(fit_data, f, indent=2)

        tmp_path.replace(fit_path)
        return True

    except Exception:
        return False


def load_default_fit(model_state, apply_excluded: bool = False) -> bool:
    """Load the generic default fit into the model state.

    This loads from default_fit.json. By default, does NOT apply the
    excluded mask since data dimensions may differ.

    Args:
        model_state: The ModelState object to load into
        apply_excluded: Whether to apply the excluded mask

    Returns:
        True if load succeeded, False otherwise
    """
    try:
        fits_folder = _get_fits_folder()
        fit_path = fits_folder / DEFAULT_FIT_FILENAME

        if not fit_path.exists():
            return False

        with fit_path.open("r", encoding="utf-8") as f:
            fit_data = json.load(f)

        return _apply_fit_state(model_state, fit_data, apply_excluded=apply_excluded)

    except Exception:
        return False


def save_fit_for_file(model_state, filepath: str) -> bool:
    """Save the current fit for a specific data file.

    This creates a fit file in the fits/ folder named after the data file.
    The fit can be restored when the same file is loaded again.

    Args:
        model_state: The ModelState object to save
        filepath: The path to the data file this fit is for

    Returns:
        True if save succeeded, False otherwise
    """
    if not filepath:
        return False

    try:
        fits_folder = _ensure_fits_folder()
        fit_data = _extract_fit_state(model_state)
        if fit_data is None:
            return False

        # Update signature with the specific file path
        fit_data["signature"]["path"] = filepath
        try:
            if os.path.isfile(filepath):
                fit_data["signature"]["size"] = os.path.getsize(filepath)
        except Exception:
            pass

        # Update source with file info
        fit_data["source"] = {
            "path": filepath,
            "name": os.path.basename(filepath),
        }
        try:
            if os.path.isfile(filepath):
                fit_data["source"]["size"] = os.path.getsize(filepath)
        except Exception:
            pass

        fit_filename = _get_fit_filename_for_file(filepath)
        fit_path = fits_folder / fit_filename
        tmp_path = fit_path.with_suffix(fit_path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(fit_data, f, indent=2)

        tmp_path.replace(fit_path)
        return True

    except Exception:
        return False


def load_fit_for_file(model_state, filepath: str, apply_excluded: bool = True) -> bool:
    """Load a saved fit for a specific data file.

    Searches for a fit file matching the given filepath and applies it
    to the model state if found.

    Args:
        model_state: The ModelState object to load into
        filepath: The path to the data file to find a fit for
        apply_excluded: Whether to apply the excluded mask

    Returns:
        True if a matching fit was found and applied, False otherwise
    """
    if not filepath:
        return False

    try:
        fits_folder = _get_fits_folder()
        if not fits_folder.exists():
            return False

        fit_filename = _get_fit_filename_for_file(filepath)
        fit_path = fits_folder / fit_filename

        if not fit_path.exists():
            return False

        with fit_path.open("r", encoding="utf-8") as f:
            fit_data = json.load(f)

        # Verify the signature matches (at least the path basename)
        sig = fit_data.get("signature", {})
        saved_path = sig.get("path")
        if saved_path:
            # Check that basenames match
            if os.path.basename(saved_path) != os.path.basename(filepath):
                return False

        return _apply_fit_state(model_state, fit_data, apply_excluded=apply_excluded)

    except Exception:
        return False


def reset_fit_for_file(filepath: str) -> bool:
    """Delete the saved fit for a specific data file.

    Args:
        filepath: The path to the data file whose fit should be deleted

    Returns:
        True if the fit file was deleted, False otherwise
    """
    if not filepath:
        return False

    try:
        fits_folder = _get_fits_folder()
        fit_filename = _get_fit_filename_for_file(filepath)
        fit_path = fits_folder / fit_filename

        if fit_path.exists():
            fit_path.unlink()
            return True
        return False

    except Exception:
        return False


def reset_default_fit() -> bool:
    """Reset/delete the default fit file.

    Returns:
        True if the default fit was deleted, False otherwise
    """
    try:
        fits_folder = _get_fits_folder()
        fit_path = fits_folder / DEFAULT_FIT_FILENAME

        if fit_path.exists():
            fit_path.unlink()
            return True
        return False

    except Exception:
        return False


def has_fit_for_file(filepath: str) -> bool:
    """Check if a saved fit exists for a specific data file.

    Args:
        filepath: The path to the data file to check

    Returns:
        True if a fit file exists, False otherwise
    """
    if not filepath:
        return False

    try:
        fits_folder = _get_fits_folder()
        if not fits_folder.exists():
            return False

        fit_filename = _get_fit_filename_for_file(filepath)
        fit_path = fits_folder / fit_filename
        return fit_path.exists()

    except Exception:
        return False
