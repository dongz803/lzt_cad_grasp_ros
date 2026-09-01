"""Resolve package resources from source and installed ROS environments."""

from __future__ import annotations

from pathlib import Path


PACKAGE_NAME = "lzt_cad_grasp_ros"


def package_share() -> Path:
    """Return the package resource directory in source or installed form."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory(PACKAGE_NAME))
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[2]


def scene_path(scene_name: str) -> Path:
    """Resolve and validate one scene filename without accepting traversal."""
    name = Path(scene_name)
    if name.name != scene_name or name.suffix.lower() != ".xml":
        raise ValueError("scene_name must be a plain XML filename")
    path = package_share() / "scenes" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def asset_path(relative_path: str) -> Path:
    """Resolve one package-owned asset while preventing directory traversal."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative_path must remain inside the assets directory")
    path = package_share() / "assets" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
