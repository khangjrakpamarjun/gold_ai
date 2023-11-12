# `reporting`

[![JFrog](https://img.shields.io/badge/JFrog-Artifact-darkgreen?style=for-the-badge)](https://mckinsey.jfrog.io/ui/packages/pypi:%2F%2Foai.reporting)

## Installation

```shell
pip install oai.reporting
```
See [package installation guide](../../../README.md) for more details

## Overview

This package contains following subpackages (see package diagram below):
* `api` – provides users with an external API (currently contains only `types` submodules)
* `charts` – provides different plots/dashboards/overviews (dictionary of charts)
* `report` – composes dictionary of charts into a standalone sharable report file
* `interactive` – provides widgets to ease charts wrangling in jupyter
* `datasets` – contains mock data for showcasing functionality
* `testing` – contains functions for testing purposes

### package diagram

![reporting package diagram](../../docs/diagrams/reporting.png)


## Usage 

Please see tutorial notebooks where we demonstrate functions usage on sample datasets:
  + [![reporting.charts.charts](https://img.shields.io/badge/TUTORIAL-charts-orange?logo=Jupyter&style=flat)](notebooks/charts/charts.ipynb) – how to plot various charts available in reporting
  + [![reporting.charts.batchplot](https://img.shields.io/badge/TUTORIAL-charts.batch__analytics-orange?logo=Jupyter&style=flat)](notebooks/charts/batchplot.ipynb) – how to plot use-case-specific charts
  + [![reporting.charts.model_performance](https://img.shields.io/badge/TUTORIAL-charts.modeling-orange?logo=Jupyter&style=flat)](notebooks/charts/model_performance.ipynb) – how to create a model performance report
  + [![reporting.report](https://img.shields.io/badge/TUTORIAL-report-orange?logo=Jupyter&style=flat)](notebooks/report.ipynb) – how to compose plots into standalone report files

## Package architecture

Please see the [API section](../../../../../docs/build/apidoc/reporting/modules.rst)
to learn more about package structure and descriptions of the functions and classes.

Before going into the details familiarize yourself with package structure below.

[UML cheat sheet](http://uml-diagrams.org).

### `reporting.api` class diagram

We try to keep all subpackages/submodules less dependent on each other.
To do so each submodule might introduce its own protocols.
And when it does, we have to make sure that it is exposed in our API
(if it's supposed to be used by other modules or users, of course).

Hence, api subpackage consolidates main external and internal types used in package.
Those types consist of model and figure types.

![reporting.api package diagram](../../docs/diagrams/api.png)

[More details about `reporting.api`](../../../../../docs/build/apidoc/reporting/reporting.api.types.rst)

### `reporting.charts` package diagram

The `charts` module contains plotting functions that produce
package compatible figures (see `api.types.PlotlyLike`, `api.MatplotlibLike`).

![reporting.charts package diagram](../../docs/diagrams/charts.png)

[More details about `reporting.charts`](../../../../../docs/build/apidoc/reporting/reporting.charts.rst)

### `reporting.report` class diagram

This subpackage implements report generation.
It is done in several steps:
* template loading; done by `report.templates`
* input rendering; done by `report.rendering`
* generation report file; done by `report.report_generation`

![reporting.report package diagram](../../docs/diagrams/report.png)

[More details about `reporting.report`](../../../../../docs/build/apidoc/reporting/reporting.report.rst)
