#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from typing import List
from setuptools import setup, find_packages

requirements: List[str] = []
setup_requirements: List[str] = []
test_requirements: List[str] = []

setup(
    author="Kapil Sachdeva",
    author_email="not@anemail.com",
    python_requires=">=3.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Utilities for Statistical Rethinking TFP Port",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description="Utilities for Statistical Rethinking TFP Port",
    include_package_data=True,
    keywords="bayesian statistics probability tensorflow tensorflow_probability",
    name="rethinking",
    packages=find_packages(include=["rethinking", "rethinking.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    version="0.1.0",
    zip_safe=False,
    package_data={},
)
