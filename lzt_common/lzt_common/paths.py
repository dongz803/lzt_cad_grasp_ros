"""Resolve package resources from source and installed ROS environments."""

from __future__ import annotations

from pathlib import Path


DESCRIPTION_PACKAGE = "lzt_description"


def package_share(package_name: str = DESCRIPTION_PACKAGE) -> Path:
    """Return one package resource directory in source or installed form."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory(package_name))
    except (ImportError, LookupError):
        candidate = Path(__file__).resolve().parents[2] / package_name
        if candidate.is_dir():
            return candidate
        raise LookupError(f"package share is unavailable: {package_name}")


def scene_path(scene_name: str) -> Path:
    """Resolve and validate one scene filename without accepting traversal."""
    name = Path(scene_name)
    if name.name != scene_name or name.suffix.lower() != ".xml":
        raise ValueError("scene_name must be a plain XML filename")
    path = package_share(DESCRIPTION_PACKAGE) / "scenes" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def asset_path(relative_path: str) -> Path:
    """Resolve one package-owned asset while preventing directory traversal."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative_path must remain inside the assets directory")
    path = package_share(DESCRIPTION_PACKAGE) / "assets" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
