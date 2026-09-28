"""Expose every source package when running the repository tests directly."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for package_root in sorted(PROJECT_ROOT.iterdir()):
    if (package_root / "package.xml").is_file():
        sys.path.insert(0, str(package_root))

