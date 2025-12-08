# models/model_elements/loader.py
"""Model element loader for YAML-based model definitions.

This module provides functionality to:
1. Discover model element YAML files in the model_elements directory
2. Parse and validate model definitions
3. Dynamically create ModelSpec classes from definitions
4. Handle errors gracefully when models are missing or invalid

The loader supports:
- Human-readable YAML format for model definitions
- Parameter metadata (type, min/max, hint, decimals, step, control)
- Mathematical function evaluation using numpy/scipy
- Graceful error handling with detailed error messages

Security:
- Expression evaluation uses a restricted sandbox with empty __builtins__
- Only pre-defined mathematical functions are available
- Dangerous patterns are blocked before evaluation
- YAML files are application-distributed, not user-uploaded content
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import yaml
import numpy as np
from scipy.special import wofz

# Set up logging for graceful error handling
logger = logging.getLogger(__name__)


class ModelElementError(Exception):
    """Exception raised when a model element cannot be loaded or is invalid."""
    pass


class ModelElementNotFoundError(ModelElementError):
    """Exception raised when a requested model element does not exist."""
    pass


class ModelElementValidationError(ModelElementError):
    """Exception raised when a model element fails validation."""
    pass


# Minimum value for FWHM parameters to prevent division by zero.
# This value is chosen to be small enough not to affect physical results
# while preventing numerical overflow in the Voigt function calculation.
_MIN_FWHM_VALUE = 1e-10


# Lazy loading pattern for base classes
# The base classes are imported from model_specs only when first needed.
# This delays the import until runtime to avoid import-time circular
# dependency issues between this module and model_specs.
_base_classes_loaded = False
_Parameter = None
_BaseModelSpec = None


def _ensure_base_classes():
    """Lazy-load base classes from model_specs.
    
    This function imports Parameter and BaseModelSpec the first time
    it's called. Using lazy loading avoids import-time circular
    dependency issues between this module and model_specs.
    """
    global _base_classes_loaded, _Parameter, _BaseModelSpec
    if not _base_classes_loaded:
        from models.model_specs import Parameter, BaseModelSpec
        _Parameter = Parameter
        _BaseModelSpec = BaseModelSpec
        _base_classes_loaded = True


# Physical constants used in model evaluation
kB = 0.086173324  # meV/K (Boltzmann constant)


# Mathematical primitive functions available for model evaluation
def _Gaussian(x, Area, Width, Center):
    """Gaussian peak using FWHM as width, integrates to Area."""
    return Area * np.sqrt(4 * np.log(2) / np.pi) / Width * np.exp(
        -4 * np.log(2) * (np.array(x) - Center) ** 2 / Width ** 2
    )


def _Lorentzian(x, Area, Width, Center):
    """Lorentzian peak integrating to Area."""
    return 2 * Area / np.pi * Width / (4 * (np.array(x) - Center) ** 2 + Width ** 2)


def _Voigt(x, Area, gauss_fwhm, lorentz_fwhm, center):
    """Voigt profile integrating to Area."""
    # Convert x to numpy array for proper broadcasting
    x = np.asarray(x, dtype=float)
    
    # Protect against division by zero with minimum values
    gauss_fwhm = max(float(gauss_fwhm), _MIN_FWHM_VALUE)
    lorentz_fwhm = max(float(lorentz_fwhm), _MIN_FWHM_VALUE)
    
    sigma = gauss_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to std
    gamma = lorentz_fwhm / 2                           # Convert FWHM to HWHM
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return Area * profile


# Available functions for evaluation expressions
_EVAL_FUNCTIONS = {
    "Gaussian": _Gaussian,
    "Lorentzian": _Lorentzian,
    "Voigt": _Voigt,
    "np": np,
    "numpy": np,
    "wofz": wofz,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "pi": np.pi,
    "kB": kB,
}


def _get_model_elements_dir() -> Path:
    """Get the path to the model_elements directory."""
    return Path(__file__).parent


def _discover_element_files() -> Dict[str, Path]:
    """Discover all .yaml model element files in the model_elements directory.
    
    Returns:
        Dict mapping element name (lowercase) to file path
    """
    elements_dir = _get_model_elements_dir()
    files = {}
    
    if not elements_dir.exists():
        logger.warning(f"Model elements directory not found: {elements_dir}")
        return files
    
    for filepath in elements_dir.glob("*.yaml"):
        # Skip files starting with underscore (private/template files)
        if filepath.name.startswith("_"):
            continue
        
        # Element name is the file name without extension
        name = filepath.stem.lower()
        files[name] = filepath
        
    return files


def _validate_element_definition(definition: Dict[str, Any], filepath: Path) -> None:
    """Validate a model element definition.
    
    Args:
        definition: The parsed YAML definition
        filepath: Path to the file (for error messages)
        
    Raises:
        ModelElementValidationError: If validation fails
    """
    # Skip saved custom models (they belong in custom_models directory, not here)
    if definition.get('category') == 'saved_custom_model':
        raise ModelElementValidationError(
            f"File '{filepath.name}' is a saved custom model and should not be in model_elements directory"
        )
    
    # Name is always required
    if "name" not in definition:
        raise ModelElementValidationError(
            f"Model element '{filepath.name}' is missing required field: name"
        )
    
    # Check if this is a composite model
    is_composite = definition.get("is_composite", False)
    
    if is_composite:
        # Composite models need components
        if "components" not in definition:
            raise ModelElementValidationError(
                f"Composite model element '{filepath.name}' is missing required field: components"
            )
        
        components = definition.get("components", [])
        if not isinstance(components, list):
            raise ModelElementValidationError(
                f"Composite model element '{filepath.name}': 'components' must be a list"
            )
        
        # Validate each component
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                raise ModelElementValidationError(
                    f"Composite model element '{filepath.name}': component {i} must be a dict"
                )
            if "element" not in comp:
                raise ModelElementValidationError(
                    f"Composite model element '{filepath.name}': component {i} is missing 'element'"
                )
    else:
        # Regular models need parameters
        if "parameters" not in definition:
            raise ModelElementValidationError(
                f"Model element '{filepath.name}' is missing required field: parameters"
            )
        
        # Validate parameters is a list
        params = definition.get("parameters", [])
        if not isinstance(params, list):
            raise ModelElementValidationError(
                f"Model element '{filepath.name}': 'parameters' must be a list"
            )
        
        # Validate each parameter has a name
        for i, param in enumerate(params):
            if not isinstance(param, dict):
                raise ModelElementValidationError(
                    f"Model element '{filepath.name}': parameter {i} must be a dict"
                )
            if "name" not in param:
                raise ModelElementValidationError(
                    f"Model element '{filepath.name}': parameter {i} is missing 'name'"
                )


def _load_element_definition(filepath: Path) -> Dict[str, Any]:
    """Load and validate a model element definition from a YAML file.
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        The validated definition dictionary
        
    Raises:
        ModelElementError: If loading or validation fails
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            definition = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ModelElementValidationError(
            f"Failed to parse YAML in '{filepath.name}': {e}"
        )
    except OSError as e:
        raise ModelElementError(
            f"Failed to read model element file '{filepath.name}': {e}"
        )
    
    if definition is None:
        raise ModelElementValidationError(
            f"Model element file '{filepath.name}' is empty"
        )
    
    _validate_element_definition(definition, filepath)
    return definition


# Whitelist of allowed AST node types for expression validation
_ALLOWED_AST_NODES: Set[type] = {
    ast.Expression,
    ast.BinOp,       # Binary operations: +, -, *, /, etc.
    ast.UnaryOp,     # Unary operations: -, +
    ast.Compare,     # Comparisons (though not typically needed)
    ast.Call,        # Function calls
    ast.Name,        # Variable names
    ast.Constant,    # Numeric/string constants
    ast.Num,         # Numbers (Python 3.7 compat)
    ast.Str,         # Strings (Python 3.7 compat)
    ast.Load,        # Load context
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,  # Operators
    ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Attribute,   # Allow numpy attribute access like np.sin
    ast.Subscript,   # Allow array indexing
    ast.Index,       # Index context (Python 3.8 compat)
    ast.Slice,       # Allow slicing
}


def _validate_expression_ast(expression: str, allowed_names: Set[str]) -> bool:
    """Validate an expression using AST analysis.
    
    Args:
        expression: The expression string to validate
        allowed_names: Set of allowed variable/function names
        
    Returns:
        True if the expression is safe, False otherwise
    """
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False
    
    # Walk the AST and check each node
    for node in ast.walk(tree):
        node_type = type(node)
        
        # Check if node type is allowed
        if node_type not in _ALLOWED_AST_NODES:
            logger.warning(f"Disallowed AST node type: {node_type.__name__}")
            return False
        
        # Check variable/function names are in whitelist
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                logger.warning(f"Disallowed name in expression: {node.id}")
                return False
        
        # Check attribute access is for allowed modules (like np.sin)
        if isinstance(node, ast.Attribute):
            # Allow accessing attributes on allowed names (e.g., np.sin, np.exp)
            if isinstance(node.value, ast.Name):
                if node.value.id not in allowed_names:
                    logger.warning(f"Disallowed attribute access: {node.value.id}.{node.attr}")
                    return False
    
    return True


def _create_evaluate_function(eval_expression: Optional[str], param_names: List[str]):
    """Create an evaluate function from a YAML expression.
    
    Args:
        eval_expression: The evaluation expression string (e.g., "Gaussian(x, Area, Width, Center)")
        param_names: List of parameter names for the model
        
    Returns:
        A function that takes (x, params_dict) and returns the evaluated result
        
    Note:
        Parameter names containing spaces are made available with underscores
        (e.g., "Gauss FWHM" becomes "Gauss_FWHM" in the expression)
        
    Security:
        - Expression is validated using AST analysis before execution
        - Only whitelisted functions and variables are allowed
        - eval() is sandboxed with empty __builtins__
        - YAML files are application-distributed, not user-uploaded
    """
    if not eval_expression:
        # Default: return zeros
        def default_evaluate(x, params):
            return np.zeros_like(np.asarray(x, dtype=float))
        return default_evaluate
    
    # Build whitelist of allowed names
    allowed_names = set(_EVAL_FUNCTIONS.keys())
    allowed_names.add('x')  # The input variable
    for name in param_names:
        allowed_names.add(name)
        # Also add underscore version
        allowed_names.add(name.replace(" ", "_"))
    
    # Validate expression using AST
    if not _validate_expression_ast(eval_expression, allowed_names):
        logger.error(f"Expression failed AST validation: {eval_expression}")
        def safe_fallback(x, params):
            return np.zeros_like(np.asarray(x, dtype=float))
        return safe_fallback
    
    def evaluate(x, params):
        """Evaluate the model expression with the given parameters."""
        try:
            # Build local namespace with parameters
            local_vars = dict(_EVAL_FUNCTIONS)
            local_vars["x"] = np.asarray(x, dtype=float)
            
            # Add parameter values to namespace
            for name in param_names:
                # Get value from params dict, handling nested dicts
                value = params.get(name, 0.0)
                if isinstance(value, dict):
                    value = value.get("value", 0.0)
                float_value = float(value) if value is not None else 0.0
                
                # Store under original name
                local_vars[name] = float_value
                # Also store with spaces replaced by underscores for easier expression writing
                safe_name = name.replace(" ", "_")
                local_vars[safe_name] = float_value
            
            # Evaluate the expression with restricted builtins
            # The expression has been validated via AST analysis above
            result = eval(eval_expression, {"__builtins__": {}}, local_vars)  # noqa: S307
            return np.asarray(result, dtype=float)
        except Exception as e:
            logger.warning(f"Error evaluating model expression: {e}")
            return np.zeros_like(np.asarray(x, dtype=float))
    
    return evaluate


def _sanitize_class_name(element_name: str) -> str:
    """Sanitize an element name to create a valid Python class name.
    
    Args:
        element_name: The element name to sanitize
        
    Returns:
        A valid Python identifier suitable for use as a class name
    """
    # Remove spaces and create valid Python identifier
    clean_name = element_name.replace(' ', '')
    # Remove any remaining invalid characters for Python identifiers
    clean_name = ''.join(c if c.isalnum() or c == '_' else '' for c in clean_name)
    # Ensure it starts with a letter or underscore
    if clean_name and not (clean_name[0].isalpha() or clean_name[0] == '_'):
        clean_name = '_' + clean_name
    # Use a default name if sanitization resulted in empty string
    if not clean_name:
        clean_name = 'Custom'
    return clean_name


def _create_composite_model_spec_class(definition: Dict[str, Any]) -> Type:
    """Create a CompositeModelSpec class from a definition dictionary.
    
    Args:
        definition: The validated definition dictionary with 'is_composite': True
        
    Returns:
        A new CompositeModelSpec subclass
    """
    _ensure_base_classes()
    
    # Import CompositeModelSpec here to avoid circular imports
    from models.model_specs import CompositeModelSpec
    
    element_name = definition.get("name", "Unknown")
    description = definition.get("description", "")
    components_def = definition.get("components", [])
    
    # Build the class
    class DynamicCompositeModelSpec(CompositeModelSpec):
        __doc__ = description or f"{element_name} composite model specification."
        _element_name = element_name
        _element_definition = definition
        
        def __init__(self):
            super().__init__()
            # Add components from definition
            for comp_def in components_def:
                element_type = comp_def.get("element")
                prefix = comp_def.get("prefix")
                default_params = comp_def.get("default_parameters", {})
                
                if not element_type:
                    continue
                
                # Convert default_parameters to the format expected by add_component
                initial_params = {}
                fixed_params = {}
                link_groups = {}
                bounds = {}
                
                for param_name, param_data in default_params.items():
                    if isinstance(param_data, dict):
                        # Extract value
                        initial_params[param_name] = param_data.get('value')
                        
                        # Track fixed state
                        if param_data.get('fixed'):
                            fixed_params[param_name] = True
                        
                        # Track link groups
                        lg = param_data.get('link_group')
                        if lg is not None and lg != 0:
                            link_groups[param_name] = lg
                        
                        # Track bounds
                        min_val = param_data.get('min')
                        max_val = param_data.get('max')
                        if min_val is not None or max_val is not None:
                            bounds[param_name] = (min_val, max_val)
                    else:
                        # Simple value
                        initial_params[param_name] = param_data
                
                try:
                    # Add the component
                    component = self.add_component(
                        element_type,
                        initial_params=initial_params,
                        prefix=prefix
                    )
                    
                    # Apply fixed state, link groups, and bounds
                    if component:
                        for param_name in component.spec.params.keys():
                            param_obj = component.spec.params[param_name]
                            
                            # Apply fixed state
                            if param_name in fixed_params:
                                param_obj.fixed = True
                            
                            # Apply link group
                            if param_name in link_groups:
                                param_obj.link_group = link_groups[param_name]
                            
                            # Apply bounds
                            if param_name in bounds:
                                min_val, max_val = bounds[param_name]
                                if min_val is not None:
                                    param_obj.min = min_val
                                if max_val is not None:
                                    param_obj.max = max_val
                        
                        # Rebuild flat params to propagate changes
                        self._rebuild_flat_params()
                        
                except Exception as e:
                    logger.warning(f"Failed to add component '{element_type}' to composite model '{element_name}': {e}")
                    continue
    
    # Give the class a meaningful name
    clean_name = _sanitize_class_name(element_name)
    class_name = f"{clean_name}ModelSpec"
    DynamicCompositeModelSpec.__name__ = class_name
    DynamicCompositeModelSpec.__qualname__ = class_name
    
    return DynamicCompositeModelSpec


def _create_model_spec_class(definition: Dict[str, Any]) -> Type:
    """Create a ModelSpec class from a definition dictionary.
    
    Args:
        definition: The validated definition dictionary
        
    Returns:
        A new ModelSpec subclass (or CompositeModelSpec if is_composite is True)
    """
    # Check if this is a composite model
    if definition.get("is_composite"):
        return _create_composite_model_spec_class(definition)
    
    _ensure_base_classes()
    
    element_name = definition.get("name", "Unknown")
    description = definition.get("description", "")
    params_def = definition.get("parameters", [])
    eval_expression = definition.get("evaluate")
    
    # Collect parameter names for evaluate function
    param_names = [p.get("name") for p in params_def if p.get("name")]
    
    # Create the evaluate function
    eval_func = _create_evaluate_function(eval_expression, param_names)
    
    # Build the class
    class DynamicModelSpec(_BaseModelSpec):
        __doc__ = description or f"{element_name} model specification."
        _element_name = element_name
        _element_definition = definition
        
        def __init__(self):
            super().__init__()
            # Add parameters from definition
            for param_def in params_def:
                name = param_def.get("name")
                if not name:
                    continue
                
                # Extract parameter attributes
                value = param_def.get("value", param_def.get("default", 0.0))
                ptype = param_def.get("type", "float")
                minimum = param_def.get("min", param_def.get("minimum"))
                maximum = param_def.get("max", param_def.get("maximum"))
                choices = param_def.get("choices")
                hint = param_def.get("hint", param_def.get("description", ""))
                decimals = param_def.get("decimals")
                step = param_def.get("step")
                control = param_def.get("control")
                fixed = param_def.get("fixed", False)
                link_group = param_def.get("link_group")
                
                param = _Parameter(
                    name=name,
                    value=value,
                    ptype=ptype,
                    minimum=minimum,
                    maximum=maximum,
                    choices=choices,
                    hint=hint,
                    decimals=decimals,
                    step=step,
                    control=control,
                    fixed=fixed,
                    link_group=link_group,
                )
                self.add(param)
        
        def evaluate(self, x, params=None):
            try:
                pvals = self.get_param_values(params)
                return eval_func(x, pvals)
            except Exception:
                return super().evaluate(x, params)
    
    # Give the class a meaningful name
    clean_name = _sanitize_class_name(element_name)
    class_name = f"{clean_name}ModelSpec"
    DynamicModelSpec.__name__ = class_name
    DynamicModelSpec.__qualname__ = class_name
    
    return DynamicModelSpec


# Cache for loaded element definitions and classes
_element_cache: Dict[str, Dict[str, Any]] = {}  # name -> definition
_class_cache: Dict[str, Type] = {}  # name -> class


def _load_all_elements() -> None:
    """Load all element definitions into the cache."""
    global _element_cache, _class_cache
    
    files = _discover_element_files()
    
    for name, filepath in files.items():
        try:
            definition = _load_element_definition(filepath)
            _element_cache[name] = definition
            
            # Also cache by the element's declared name
            declared_name = definition.get("name", "").lower()
            if declared_name and declared_name != name:
                _element_cache[declared_name] = definition
                
        except ModelElementError as e:
            logger.warning(f"Failed to load model element '{name}': {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading model element '{name}': {e}")


def reload_elements() -> None:
    """Reload all model element definitions from disk.
    
    This clears the cache and re-discovers all element files.
    Useful for development or when element files are modified.
    """
    global _element_cache, _class_cache
    _element_cache.clear()
    _class_cache.clear()
    _load_all_elements()


def list_available_elements() -> List[str]:
    """List all available model element names.
    
    Returns:
        List of element names (display names from definitions)
    """
    if not _element_cache:
        _load_all_elements()
    
    # Return unique display names
    names = set()
    for definition in _element_cache.values():
        name = definition.get("name")
        if name:
            names.add(name)
    
    return sorted(names)


def _normalize_element_name(name: str) -> str:
    """Normalize an element name for cache lookup.
    
    Converts to lowercase, strips whitespace, and removes common suffixes
    like 'modelspec' and 'model' to allow flexible name matching.
    
    Args:
        name: The element name to normalize
        
    Returns:
        Normalized lowercase name suitable for cache lookup
    """
    normalized = (name or "").strip().lower()
    
    # Remove common suffixes for flexible matching
    if normalized not in _element_cache:
        normalized = normalized.replace("modelspec", "").replace("model", "").strip()
    
    return normalized


def get_element_spec(element_name: str):
    """Get a model element specification instance by name.
    
    This returns a new instance of the ModelSpec for the requested element.
    
    Args:
        element_name: The name of the element (case-insensitive)
        
    Returns:
        A ModelSpec instance for the element
        
    Raises:
        ModelElementNotFoundError: If the element is not found
    """
    if not _element_cache:
        _load_all_elements()
    
    # Normalize name for lookup
    name_lower = _normalize_element_name(element_name)
    
    if name_lower not in _element_cache:
        available = list_available_elements()
        raise ModelElementNotFoundError(
            f"Model element '{element_name}' not found. "
            f"Available elements: {', '.join(available) if available else 'none'}"
        )
    
    # Get or create the class
    spec_class = get_element_spec_class(element_name)
    return spec_class()


def get_element_spec_class(element_name: str) -> Type:
    """Get the ModelSpec class for a model element.
    
    This returns the class, not an instance. Useful for introspection
    or when you need to create multiple instances.
    
    Args:
        element_name: The name of the element (case-insensitive)
        
    Returns:
        The ModelSpec class for the element
        
    Raises:
        ModelElementNotFoundError: If the element is not found
    """
    if not _element_cache:
        _load_all_elements()
    
    # Normalize name for lookup
    name_lower = _normalize_element_name(element_name)
    
    if name_lower not in _element_cache:
        available = list_available_elements()
        raise ModelElementNotFoundError(
            f"Model element '{element_name}' not found. "
            f"Available elements: {', '.join(available) if available else 'none'}"
        )
    
    # Check class cache
    if name_lower in _class_cache:
        return _class_cache[name_lower]
    
    # Create the class
    definition = _element_cache[name_lower]
    spec_class = _create_model_spec_class(definition)
    
    # Cache it
    _class_cache[name_lower] = spec_class
    
    # Also cache by declared name
    declared_name = definition.get("name", "").lower()
    if declared_name and declared_name != name_lower:
        _class_cache[declared_name] = spec_class
    
    return spec_class
