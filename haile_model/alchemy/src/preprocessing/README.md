# `preprocessing`

[![Tutorial notebook](https://img.shields.io/badge/jupyter-tutorial_notebook-orange?style=for-the-badge&logo=jupyter)](notebooks/preprocessing.ipynb)
[![JFrog](https://img.shields.io/badge/JFrog-Artifact-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.feature_factory)

## Installation
```shell
pip install oai.preprocessing
```
See [package installation guide](../../../README.md) for more details

## Overview
`preprocessing` package contains functions to help refine raw timeseries data. 

## Usage
This package is kedro independent, meaning every function or class in this package can 
be imported and used for their own purposes.

It requires data in `pandas.DataFrame` and `TagDict` provided in `optimus_core` package. We provide `sample_input_data.csv` and `sample_tag_dict.csv` as references.
<br>

Please see:
- [tutorial notebook](notebooks/preprocessing.ipynb) to learn more about basic package workflow
- [API section](../../../../docs/build/html/docs/build/apidoc/preprocessing/modules.html) to learn more about package structure and descriptions of the functions
<br><br>
