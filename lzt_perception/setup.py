from pathlib import Path
from setuptools import find_packages, setup

name = "lzt_perception"
root = Path(__file__).parent
data = [("share/ament_index/resource_index/packages", ["resource/" + name]), ("share/" + name, ["package.xml"])]
for path in sorted((root / "config").rglob("*")):
    if path.is_file():
        relative = path.relative_to(root)
        data.append((f"share/{name}/{relative.parent}", [str(relative)]))
setup(name=name, version="0.2.0", packages=find_packages(), data_files=data,
      install_requires=["setuptools", "numpy"], zip_safe=False,
      maintainer="lzt", maintainer_email="lzt@todo.todo",
      description="6D pose perception and reliability research.", license="Apache-2.0")

