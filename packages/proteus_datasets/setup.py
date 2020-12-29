#!/usr/bin/env python

"""The setup script."""

from setuptools import find_namespace_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["pycocotools==2.0.2"]

test_requirements = ["pytest>=3"]

setup(
    name="proteus.datasets",
    version="0.0.1",
    description="Proteus datasets",
    author="Pieter Blomme",
    author_email="pieter.blomme@gmail.com",
    python_requires=">=3.6",
    classifiers=[],
    keywords="",
    entry_points={},
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    namespace_packages=["proteus"],
    packages=find_namespace_packages(exclude=["tests"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    zip_safe=False,
)
