from setuptools import find_packages, setup

with open("README.md") as file:
    read_me_description = file.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ism_model",
    version="1.0",
    description="ISM model library",
    author="Nikita Denisov",
    packages=find_packages(where="."),
    long_description=read_me_description,
    install_requires=required,
)