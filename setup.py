import os
from setuptools import setup, find_packages


# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    "pytorch-lightning>=1.5.10",
    "torch>=1.10.0",
    "torchmetrics>=0.8.0",
    "torchvision>=0.10.0",
    "scikit-learn>=0.24.2",
    "numpy>=1.19.5",
    "matplotlib>=3.3.4",
    "seaborn>=0.11.2",
    "ConfigSpace>=0.4.0"
]

setup(
    name="minimal-bayesopt",
    author="Stefan Werner",
    description=(
        "A minimal example of using Bayesian Optimization to tune the learning"
        " rate of a ResNet on FashionMNIST."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stwerner97/minimal-bayesopt",
    packages=find_packages(),
    python_requires=">=3.6.*, <3.9.*",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
