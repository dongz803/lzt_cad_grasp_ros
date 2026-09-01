from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_NAME = "lzt_cad_grasp_ros"
ROOT = Path(__file__).parent


def project_data_files() -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = [
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md", "requirements.txt"]),
    ]
    for directory in ("assets", "config", "launch", "scenes"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                relative = path.relative_to(ROOT)
                entries.append(
                    (f"share/{PACKAGE_NAME}/{relative.parent}", [str(relative)])
                )
    return entries


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=project_data_files(),
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=False,
    maintainer="lzt",
    maintainer_email="lzt@todo.todo",
    description="CAD-driven learned pose and robust industrial pick-place exploration",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cad_grasp_exploration_node = "
            "lzt_cad_grasp_ros.exploration_node:main",
        ],
    },
)
