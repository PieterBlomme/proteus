#!/usr/bin/env python

"""The setup script."""

from setuptools import find_namespace_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "opencv-python-headless==4.4.0.40",
    "requests==2.25.1",
    "onnx==1.8.0",
    "onnxruntime==1.6.0",
    "jinja2==2.11.2",
    "pydantic==1.5.1",
    "tritonclient[http]==2.3.0",
]

test_requirements = ["pytest>=3"]

setup(
    name="proteus.models.base",
    version="0.0.1",
    description="Proteus models",
    author="Pieter Blomme",
    author_email="pieter.blomme@gmail.com",
    python_requires=">=3.6",
    classifiers=[],
    keywords="",
    entry_points={},
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    namespace_packages=["proteus.models"],
    packages=find_namespace_packages(exclude=["tests"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    zip_safe=False,
)
