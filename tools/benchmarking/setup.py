#!/usr/bin/env python

"""The setup script."""

from setuptools import find_namespace_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "proteus.datasets==0.0.1",
    "requests==2.25.1",
    "jinja2==2.11.2",
    "pandas==1.2.0",
    "tabulate==0.8.7",
]
test_requirements = []

setup(
    name="proteus.tools.benchmarking",
    version="0.0.1",
    description="Proteus benchmarking tool",
    author="Pieter Blomme",
    author_email="pieter.blomme@gmail.com",
    python_requires=">=3.6",
    classifiers=[],
    keywords="",
    entry_points={
        "console_scripts": [
            "proteus.benchmark=proteus.tools.benchmarking.suite:main",
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
