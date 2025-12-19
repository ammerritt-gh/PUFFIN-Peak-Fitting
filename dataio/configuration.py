from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Any

@dataclass
class Config:
    last_loaded_file: Optional[str] = None
    default_load_folder: str = ""
    default_save_folder: str = ""
    config_folder: str = ""
    config_filename: str = "settings.json"
    # persisted queued files (list of dicts with keys 'path' and optional 'name')
    queued_files: List[dict] = field(default_factory=list)
    # index of active queued file, or None
    queued_active: Optional[int] = None
    # save dialog preferences
    save_delimiter: str = "comma"  # "comma", "tab", or "space"

    def __post_init__(self):
        # ensure folders are normalized strings
        self.default_load_folder = str(self.default_load_folder or "")
        self.default_save_folder = str(self.default_save_folder or "")
        self.config_folder = str(self.config_folder or "")

    @property
    def config_path(self) -> Path:
        return Path(self.config_folder) / self.config_filename

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self) -> None:
        cfg_path = self.config_path
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp.replace(cfg_path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        if path is None:
            raise ValueError("path must be provided for load()")
        path = Path(path)
        if not path.exists():
            # return default config with folder set
            return cls(config_folder=str(path.parent), config_filename=path.name)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                last_loaded_file=data.get("last_loaded_file"),
                default_load_folder=data.get("default_load_folder", ""),
                default_save_folder=data.get("default_save_folder", ""),
                config_folder=str(path.parent),
                config_filename=path.name,
                queued_files=data.get("queued_files", []),
                queued_active=data.get("queued_active", None),
                save_delimiter=data.get("save_delimiter", "comma"),
            )
        except Exception:
            # on parse error return defaults and keep config folder
            return cls(config_folder=str(path.parent), config_filename=path.name)

# Module-level singleton accessor
_config_singleton: Optional[Config] = None

def _default_repo_config_folder() -> Path:
    # repo root is two levels up from this file: ...\BigFit\dataio\configuration.py
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "config"

def get_config(recreate: bool = False) -> Config:
    """
    Return a singleton Config instance.
    On first call the JSON file in the repo config folder is loaded (or created).
    Set recreate=True to reload from disk.
    """
    global _config_singleton
    if _config_singleton is not None and not recreate:
        return _config_singleton

    cfg_folder = _default_repo_config_folder()
    cfg_file = cfg_folder / "settings.json"
    if cfg_file.exists():
        cfg = Config.load(cfg_file)
    else:
        cfg = Config(
            last_loaded_file=None,
            default_load_folder=str(Path.home()),
            default_save_folder=str(Path.home()),
            config_folder=str(cfg_folder),
            config_filename="settings.json",
        )
        cfg.save()
    _config_singleton = cfg
    return _config_singleton
