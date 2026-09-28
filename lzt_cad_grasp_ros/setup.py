from setuptools import find_packages, setup


PACKAGE_NAME = "lzt_cad_grasp_ros"
setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=False,
    maintainer="lzt",
    maintainer_email="lzt@todo.todo",
    description="Compatibility facade for the LZT robot algorithm stack",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cad_grasp_exploration_node = "
            "lzt_cad_grasp_ros.exploration_node:main",
        ],
    },
)
