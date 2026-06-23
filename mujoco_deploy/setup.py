from setuptools import find_packages
from distutils.core import setup

setup(
    name="mujoco_deploy",
    version="1.0.0",
    author="Mondo Robotics",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="haha@example.com",
    description="haha",
    install_requires=[
        "matplotlib",
        "pyyaml",
        # "onnx==1.20.0",
        # "onnxruntime==1.23.2",
        # "mujoco==3.2.7",
        "opencv-python-headless",
        # "protobuf",
        # "pybind11",
        # "pyzmq",
    ],
)
