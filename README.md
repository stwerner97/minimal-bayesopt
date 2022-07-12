[![CI](https://github.com/stwerner97/minimal-bayesopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/stwerner97/minimal-bayesopt/actions/workflows/python-package.yml) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stwerner97/minimal-bayesopt/blob/main/bayesopt.ipynb)

# Minimal-BayesOpt
A minimal example of using Bayesian Optimization to tune the learning rate of a ResNet on FashionMNIST.

The contents of this repository can be run in [Google Colaboratory](https://colab.research.google.com/github/stwerner97/minimal-bayesopt/blob/main/bayesopt.ipynb) or by cloning the repository.

<p align="center">
    <img src="https://user-images.githubusercontent.com/36734964/178485012-20326fd1-47e8-4c3d-bc00-52ccff53ead3.gif" width="85%"/>
</p>


Each plot shows the parameters sampled so far and their objective value (F1 validation score). The cross highlights the best performing parameter. The black line and orange area around it show the posterior mean and uncertainty estimates, respectively. Additionally, we show functions samples from the posterior in grey. Below, the acquisition function landscape (here: expected improvement) is shown.

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
