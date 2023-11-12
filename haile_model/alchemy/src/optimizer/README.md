# `optimizer`
[![Tutorial notebook](https://img.shields.io/badge/jupyter-tutorial_notebook-orange?style=for-the-badge&logo=jupyter)](../../docs/source/03_tutorial/01_tutorial.ipynb)
[![JFrog](https://img.shields.io/badge/JFrog-Artifact-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.optimizer)

## Installation
```shell
pip install oai.optimizer
```
See [package installation guide](../../../README.md) for more details

## Overview
`optimizer` is one of the core package in OptimusAI. It contains critical components to run optimization.

## Usage
Core components in `optimizer` are: 
- [`solver`](../../docs/source/04_user_guide/03_solver.md)
- [`repair`](../../docs/source/04_user_guide/02_repair.ipynb)
- [`penalty`](../../docs/source/04_user_guide/01_penalty.ipynb)
- [`stopper`](../../docs/source/04_user_guide/05_stopper.md)

`optimizer` package is a kedro-agnostic package so we don't provide kedro pipeline.
<br><br>


```{eval-rst}
.. toctree::
   :maxdepth: 1

   ../../docs/source/03_tutorial/tutorial.rst
   ../../docs/source/04_user_guide/user_guide.rst
   ../../docs/source/05_examples/examples.rst
   ../../docs/source/02_faq/01_faq
```