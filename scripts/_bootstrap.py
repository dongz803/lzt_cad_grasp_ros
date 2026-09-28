"""Source-tree import bootstrap shared by developer command-line scripts."""

from pathlib import Path
import sys


def activate_project_packages(project_root: Path) -> None:
    for package_root in sorted(project_root.iterdir()):
        if (package_root / "package.xml").is_file():
            sys.path.insert(0, str(package_root))

