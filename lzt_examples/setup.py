from setuptools import find_packages, setup
name = "lzt_examples"
setup(name=name, version="0.2.0", packages=find_packages(), data_files=[("share/ament_index/resource_index/packages", ["resource/" + name]), ("share/" + name, ["package.xml"])], install_requires=["setuptools"], zip_safe=False, maintainer="lzt", maintainer_email="lzt@todo.todo", description="Robot-stack examples.", license="Apache-2.0", entry_points={"console_scripts": ["show_robot_stack = lzt_examples.show_robot_stack:main"]})

