"""
Instrument configuration loading and management.

This module provides functionality to load instrument definitions from YAML files
and manage instrument state during analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class CrystalSpec:
    """Specification for a monochromator or analyzer crystal."""
    name: str
    d_spacing: float  # Angstroms
    mosaic: float = 0.4  # degrees


@dataclass
class CollimatorPosition:
    """Specification for a collimator position."""
    name: str
    label: str
    options: List[Any]  # Can be numbers or strings like "open"


@dataclass
class ModuleParameter:
    """Parameter for an experimental module."""
    name: str
    label: str
    type: str  # "float", "int", "bool", "str", "choice"
    min: Optional[float] = None
    max: Optional[float] = None
    default: Any = None
    choices: Optional[List[str]] = None


@dataclass
class ExperimentalModule:
    """Specification for an experimental module (e.g., cryostat, furnace)."""
    name: str
    label: str
    enabled: bool = False
    parameters: List[ModuleParameter] = field(default_factory=list)


@dataclass
class SlitDimension:
    """Specification for a single slit dimension (horizontal or vertical)."""
    name: str
    label: str
    min: float
    max: float
    default: float
    step: float = 0.5


@dataclass
class SlitPosition:
    """Specification for a slit position with multiple dimensions."""
    name: str
    label: str
    dimensions: List[SlitDimension] = field(default_factory=list)


@dataclass
class ComponentSpec:
    """Specification for monochromator or analyzer."""
    crystals: List[CrystalSpec] = field(default_factory=list)
    focusing: List[str] = field(default_factory=list)


@dataclass
class InstrumentConfig:
    """
    Complete instrument configuration.
    
    Contains all specifications for an instrument including arm lengths,
    crystal options, collimators, slits, and experimental modules.
    """
    name: str
    type: str
    description: str = ""
    arm_lengths: Dict[str, float] = field(default_factory=dict)
    monochromator: Optional[ComponentSpec] = None
    analyzer: Optional[ComponentSpec] = None
    collimators: List[CollimatorPosition] = field(default_factory=list)
    modules: List[ExperimentalModule] = field(default_factory=list)
    slits: List[SlitPosition] = field(default_factory=list)
    
    def get_arm_length(self, key: str, default: float = 1000.0) -> float:
        """Get arm length by key with default fallback."""
        return self.arm_lengths.get(key, default)
    
    def get_monochromator_crystals(self) -> List[CrystalSpec]:
        """Get list of available monochromator crystals."""
        if self.monochromator:
            return self.monochromator.crystals
        return []
    
    def get_analyzer_crystals(self) -> List[CrystalSpec]:
        """Get list of available analyzer crystals."""
        if self.analyzer:
            return self.analyzer.crystals
        return []
    
    def get_monochromator_focusing_options(self) -> List[str]:
        """Get list of monochromator focusing options."""
        if self.monochromator:
            return self.monochromator.focusing
        return []
    
    def get_analyzer_focusing_options(self) -> List[str]:
        """Get list of analyzer focusing options."""
        if self.analyzer:
            return self.analyzer.focusing
        return []


def _parse_crystal_specs(data: List[Dict[str, Any]]) -> List[CrystalSpec]:
    """Parse crystal specifications from YAML data."""
    crystals = []
    for item in data:
        crystals.append(CrystalSpec(
            name=item.get("name", "Unknown"),
            d_spacing=float(item.get("d_spacing", 3.354)),
            mosaic=float(item.get("mosaic", 0.4))
        ))
    return crystals


def _parse_component_spec(data: Dict[str, Any]) -> ComponentSpec:
    """Parse component (monochromator/analyzer) specification from YAML data."""
    crystals = _parse_crystal_specs(data.get("crystals", []))
    focusing = data.get("focusing", [])
    return ComponentSpec(crystals=crystals, focusing=focusing)


def _parse_collimator_positions(data: Dict[str, Any]) -> List[CollimatorPosition]:
    """Parse collimator positions from YAML data."""
    positions = []
    for item in data.get("positions", []):
        positions.append(CollimatorPosition(
            name=item.get("name", ""),
            label=item.get("label", ""),
            options=item.get("options", [])
        ))
    return positions


def _parse_module_parameter(data: Dict[str, Any]) -> ModuleParameter:
    """Parse module parameter from YAML data."""
    return ModuleParameter(
        name=data.get("name", ""),
        label=data.get("label", ""),
        type=data.get("type", "float"),
        min=data.get("min"),
        max=data.get("max"),
        default=data.get("default"),
        choices=data.get("choices")
    )


def _parse_experimental_modules(data: List[Dict[str, Any]]) -> List[ExperimentalModule]:
    """Parse experimental modules from YAML data."""
    modules = []
    for item in data:
        params = [_parse_module_parameter(p) for p in item.get("parameters", [])]
        modules.append(ExperimentalModule(
            name=item.get("name", ""),
            label=item.get("label", ""),
            enabled=item.get("enabled", False),
            parameters=params
        ))
    return modules


def _parse_slit_dimension(data: Dict[str, Any]) -> SlitDimension:
    """Parse slit dimension from YAML data."""
    return SlitDimension(
        name=data.get("name", ""),
        label=data.get("label", ""),
        min=float(data.get("min", 0.0)),
        max=float(data.get("max", 50.0)),
        default=float(data.get("default", 10.0)),
        step=float(data.get("step", 0.5))
    )


def _parse_slit_positions(data: Dict[str, Any]) -> List[SlitPosition]:
    """Parse slit positions from YAML data."""
    positions = []
    for item in data.get("positions", []):
        dims = [_parse_slit_dimension(d) for d in item.get("dimensions", [])]
        positions.append(SlitPosition(
            name=item.get("name", ""),
            label=item.get("label", ""),
            dimensions=dims
        ))
    return positions


def load_instrument_config(path: Path) -> InstrumentConfig:
    """
    Load instrument configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        InstrumentConfig object with parsed configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Instrument configuration not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
    
    if not isinstance(data, dict):
        raise ValueError("Invalid instrument configuration: expected dictionary")
    
    # Parse components
    monochromator = None
    if "monochromator" in data:
        monochromator = _parse_component_spec(data["monochromator"])
    
    analyzer = None
    if "analyzer" in data:
        analyzer = _parse_component_spec(data["analyzer"])
    
    collimators = []
    if "collimators" in data:
        collimators = _parse_collimator_positions(data["collimators"])
    
    modules = _parse_experimental_modules(data.get("modules", []))
    
    slits = []
    if "slits" in data:
        slits = _parse_slit_positions(data["slits"])
    
    return InstrumentConfig(
        name=data.get("name", "Unknown Instrument"),
        type=data.get("type", "generic"),
        description=data.get("description", ""),
        arm_lengths=data.get("arm_lengths", {}),
        monochromator=monochromator,
        analyzer=analyzer,
        collimators=collimators,
        modules=modules,
        slits=slits
    )


def list_available_instruments(config_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    List all available instrument configurations.
    
    Args:
        config_dir: Directory containing instrument configs (default: config/instruments)
        
    Returns:
        List of dicts with 'name', 'path', and 'description' keys
    """
    if config_dir is None:
        # Default to repo config/instruments directory
        repo_root = Path(__file__).resolve().parent.parent
        config_dir = repo_root / "config" / "instruments"
    
    if not config_dir.exists():
        return []
    
    instruments = []
    for yaml_file in config_dir.glob("*.yaml"):
        try:
            config = load_instrument_config(yaml_file)
            instruments.append({
                'name': config.name,
                'path': str(yaml_file),
                'description': config.description,
                'type': config.type
            })
        except Exception:
            # Skip invalid configurations
            continue
    
    return instruments


def get_default_instrument_path() -> Optional[Path]:
    """Get the path to the default instrument configuration."""
    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "config" / "instruments"
    
    # Try to find a default instrument (first YAML file)
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml"))
        if yaml_files:
            return yaml_files[0]
    
    return None
