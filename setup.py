#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import subprocess
from subprocess import PIPE

# get package version
version = ""
with open(os.path.join("TimeSeriesDL", "version.txt"), "r", encoding="UTF-8") as file:
    version = file.read().strip()

# get readme content
readme = ""
with open("README.md", "r", encoding="UTF-8") as file:
    readme = file.read()

# get requirements from conda environment
os.system("pip freeze > requirements.txt")
conda_list = subprocess.run(['conda', 'list'], stdout=PIPE, stderr=PIPE, universal_newlines=True)

requirements = []
with open("requirements.txt", "r", encoding="UTF-8") as f:
    for line in f.readlines():
        if line.split("==")[0] in conda_list.stdout:
            requirements.append(line.strip())
        elif line.split(" @")[0] in conda_list.stdout:
            requirements.append(line.split(" @")[0].strip())

with open("requirements.txt", "w", encoding="UTF-8") as f:
    for line in requirements:
        f.write(line + "\n")

# run the setup
setup(
    author="Huntler",
    python_requires=">=3.9",
    classifiers=[],
    description="Deep learning framework to predict time series.",
    install_requirements=requirements,
    keywords="Deep Learning,Time Series,TimeSeriesDL",
    name="TimeSeriesDL",
    packages=find_packages(include=[
        "TimeSeriesDL", "TimeSeriesDL.*"
    ]),
    package_dir={"TimeSeriesDL": "TimeSeriesDL"},
    include_package_data=True,
    url="https://github.com/Huntler/TimeSeriesDL",
    version=version,
    zip_safe=True
)
