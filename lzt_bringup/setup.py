from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

name = "lzt_bringup"
root = Path(__file__).parent
data = [("share/ament_index/resource_index/packages", ["resource/" + name]), ("share/" + name, ["package.xml"]), ("share/" + name + "/launch", glob("launch/*.launch.py"))]
for path in sorted((root / "config").rglob("*")):
    if path.is_file():
        relative = path.relative_to(root)
        data.append((f"share/{name}/{relative.parent}", [str(relative)]))
setup(name=name, version="0.2.0", packages=find_packages(), data_files=data,
      install_requires=["setuptools"], zip_safe=False,
      maintainer="lzt", maintainer_email="lzt@todo.todo",
      description="System launch and configuration.", license="Apache-2.0",
      entry_points={"console_scripts": ["exploration_node = lzt_bringup.exploration_node:main"]})

