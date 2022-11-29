#!/usr/bin/env python3
import os
import sys
import setuptools
from distutils.core import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="linastt",
    version="1.0",
    description="Tools to experiment with speech recognition, using several Python libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jérôme Louradour",
    author_email="jlouradour@linagora.com",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    package_data={"linastt": []},
    install_requires=[
        # TODO
    ],
    python_requires=">=3.7",
    url="https://github.com/linto-ai/stt-end2end-expes",
)
