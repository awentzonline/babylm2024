#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="babylm2024",
    version="0.0.1",
    description="Entry for BabyLM Challenge 2024",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    url="https://github.com/awentzonline/babylm2024",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = babylm.train:main",
            "eval_command = babylm.eval:main",
        ]
    },
)
