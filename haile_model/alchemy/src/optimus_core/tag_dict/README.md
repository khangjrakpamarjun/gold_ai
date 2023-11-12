# `tag_dict`
[![Tutorial notebook](https://img.shields.io/badge/jupyter-tutorial_notebook-orange?style=for-the-badge&logo=jupyter)](../notebooks/tag_dict.ipynb)
## Overview
The Tag dictionary (`TagDict`) is the central OptimusAI utility for managing tags. It is an important tool for communicating with subject-matter experts as it captures all tag meta-data.

> Tag is a unique identifier of a sensor used in [PI System](https://www.osisoft.com/pi-system). Usually it is a string containing letters and numbers. Examples: `CDEP158`, `CDM158`

```{note}
As a general rule, we suggest that you use the tag dictionary to store all information about an **individual** tag such as the expected range, the clear name, or a mapping to one or multiple models. You should place higher level parameterization (e.g. per-dataset) into project parameters section (e.g. Kedro pipeline's `conf` section) instead.
```

Learn more about subpackage structure, function and class interfaces in the [API section](../../../../../docs/build/apidoc/optimus_core/optimus_core.tag_dict.rst).  

## Key columns
We have designed the tag dictionary to allow you to add any columns you want. Add a column to the underlying CSV to see it reflected in the `TagDict` object. A small number of columns are required to ensure proper function and are validated whenever a `TagDict` object is created or loaded from a CSV file.

The minimum columns required to construct an instance of `TagDict` are:

| column                | description                                                                                                                                           |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tag`                 | tag name (key)                                                                                                                                        |
| `name`                | human-readable clear name                                                                                                                             |
| `tag_type`            | functional type. One of {`input`, `output`, `state`, `control`, `on_off`}                                                                             |
| `data_type`           | data type. One of {`numeric`, `categorical`, `boolean`, `datetime`}                                                                                   |
| `unit `               | unit of measurement                                                                                                                                   |
| `range_min`           | lower limit of tag range (values that the measurement can physically take)                                                                            |
| `range_max`           | upper limit of tag range (values that the measurement can physically take)                                                                            |
| `on_off_dependencies` | names (keys) of on/off tags which determine the current tag's on/off state. If one of the dependencies is off, the current tag is considered off, too |
| `derived`             | indicates whether a tag is an original sensor reading or artificially created / derived                                                               |

## Extra columns

There are some columns which are not mandatory in the tag dictionary, but which many OAI components and shared code commonly use. You should consider using these columns before inventing new column names, so that you maintain compatibility with other Optimus solutions.

The common extra columns used in an instance of `TagDict` are:

| column              | description                                                                              |
|---------------------|------------------------------------------------------------------------------------------|
| `area`              | plant area                                                                               |
| `sub_area`          | plant sub-area                                                                           |
| `op_min`            | lower limit of operating range (values that should be considered for a control variable) |
| `op_max`            | upper limit of operating range (values that should be considered for a control variable) |
| `max_delta`         | maximum change from current value allowed during optimization                            |
| `constraint_set`    | set of permissible values for control                                                    |
| `agg_window_length` | length of window over which to aggregate during static feature creation                  |
| `agg_method`        | static feature creation aggregation method                                               |
| `notes`             | free-text notes                                                                          |
| `model_feature`     | indicates tag as a feature of the model                                                  |
| `model_target`      | indicates tag as the model target                                                        |
