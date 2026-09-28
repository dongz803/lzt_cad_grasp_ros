from setuptools import find_packages, setup
name = "lzt_benchmark"
setup(name=name, version="0.2.0", packages=find_packages(), data_files=[("share/ament_index/resource_index/packages", ["resource/" + name]), ("share/" + name, ["package.xml"])], install_requires=["setuptools", "numpy"], zip_safe=False, maintainer="lzt", maintainer_email="lzt@todo.todo", description="Benchmarks and evaluation.", license="Apache-2.0")

