from setuptools import setup

package_name = "tello_gate"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"replay = {package_name}.replay_node:main",
            f"buffer = {package_name}.buffer_node:main",
            f"inference = {package_name}.inference_node:main",
        ],
    },
)
