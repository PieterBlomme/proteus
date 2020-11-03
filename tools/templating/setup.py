#!/usr/bin/env python

"""The setup script."""

from setuptools import find_namespace_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["cookiecutter==1.7.2"]
test_requirements = []

setup(
    name="proteus.tools.templating",
    version="0.0.1",
    description="Proteus RetinaNet",
    author="Pieter Blomme",
    author_email="pieter.blomme@gmail.com",
    python_requires=">=3.6",
    classifiers=[],
    keywords="",
    entry_points={
        "console_scripts": [
            "proteus.template=proteus.tools.templating.command_line:main"
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    namespace_packages=["proteus.tools"],
    packages=find_namespace_packages(exclude=["tests"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    zip_safe=False,
)
