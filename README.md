[![CI](https://github.com/stwerner97/minimal-bayesopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/stwerner97/minimal-bayesopt/actions/workflows/python-package.yml) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stwerner97/minimal-bayesopt/blob/main/bayesopt.ipynb)

# Minimal-BayesOpt
A minimal example of using Bayesian Optimization to tune the learning rate of a ResNet on FashionMNIST.

The contents of this repository can be run in [Google Colaboratory](https://colab.research.google.com/github/stwerner97/minimal-bayesopt/blob/main/bayesopt.ipynb) or by cloning the repository.

## Installation
### From Source
The simplest option is to install the dependencies of `minimal-bayesopt` using pip:
```bash
pip install -e .
```
This is equivalent to running `pip install -r requirements.txt`.

## Example Usage
The Bayesian Optimization can be run via `script.py`:
```bash
python script.py --niters 10 --max_epochs 20 --num_gpus 1 --output bayesopt.pdf
```
For additional options, please check the `script.py` file.
