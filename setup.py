#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


with open("README.md") as readme:
    long_description = readme.read()

setuptools.setup(
    name="i3deepice",
    version="0.1",
    author="Theo Glauch",
    author_email="theo.glauch@tum.de",
    description="DeepIceLearning Icetray Module",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/tglauch/i3deepice/",
    project_urls={
        "Source": "https://github.com/tglauch/i3deepice/",
        "Tracker": "https://github.com/tglauch/i3deepice/issues/"
        },
    license="GPLv3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering"
        ],
    python_requires=">=2.7",
    install_requires=[
        'tensorflow>=1.5',
        'keras>=2.0',
        'numpy>1.14'],
    packages = ['i3deepice']
    )
