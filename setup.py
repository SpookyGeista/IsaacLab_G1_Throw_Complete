#!/usr/bin/env python3

# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Setup script for the extension."""

from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of the extension.toml file
import os

# Fallback configuration if extension.toml doesn't exist or can't be read
extension_toml = {
    "package": {
        "title": "Unitree Sim IsaacLab",
        "description": "Unitree G1 robot simulation tasks for Isaac Lab",
        "version": "0.1.0",
    }
}

setup(
    name="unitree_sim_isaaclab",
    version=extension_toml["package"]["version"],
    author="Unitree Robotics",
    description=extension_toml["package"]["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    python_requires=">=3.8",
    install_requires=[
        "isaaclab",
    ],
    extras_require={
        "dev": [
            "pre-commit",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
