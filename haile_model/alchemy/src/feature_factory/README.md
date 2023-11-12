# `feature_factory`

[![Tutorial notebook](https://img.shields.io/badge/jupyter-tutorial_notebook-orange?style=for-the-badge&logo=jupyter)](notebooks/feature_factory.ipynb)
[![JFrog](https://img.shields.io/badge/JFrog-Artifact-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.feature-factory)

## Installation
```shell
pip install oai.feature_factory
```
See [package installation guide](../../../README.md) for more details

## Overview
`feature_factory` package provides an API for producing derived features, and visualizing its dependency graphs.


## Usage 

This package is kedro independent, meaning every function or class in this package can be imported and used for their own purposes. 

`feature_factory` expects a wide-format dataframe of sensor time-series data in `pandas.DataFrame`. We provide `sample_preprocessed_data.csv` as a reference.


Please see 
- [tutorial notebook](notebooks/feature_factory.ipynb) to learn more about basic package workflow
- [API section](../../../../../docs/build/apidoc/feature_factory/modules.rst) to learn more about package structure and descriptions of the functions


## Additional information
[`FeatureFactory`](build/apidoc/feature_factory/feature_factory.nodes.html#feature_factory.nodes.feature_factory.FeatureFactory) is a core concept of `feature_factory` package, which is a subclass of [`optimus_core.transformer.Transformer`](../../../optimus_core/src/optimus_core/transformer/README.md).
It is highly recommended to learn what [`optimus_core.transformer.Transformer`](../../../optimus_core/src/optimus_core/transformer/README.md) is and why we use it before using this package. 
