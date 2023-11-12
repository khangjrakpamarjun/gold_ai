# `transformer`

## Overview
Transformers are [`sklearn` classes](https://scikit-learn.org/stable/data_transforms.html) that transform the input data. They are represented by classes that have the following methods:

* `fit` (to learn the model parameters)
* `transform` (to transform the input data)

In OptimusAI, we chain transformers for dynamic calculation of features that are not conducted in the static steps. Transformers that may be used include:

* Imputers
* PCA
* Feature Selection
* [One-hot Encoding](https://en.wikipedia.org/wiki/One-hot)
* ...

Transformers are used in combination with [`sklearn` pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

Learn more about subpackage structure, function and class interfaces in the [API section](../../../../../docs/build/apidoc/optimus_core/optimus_core.transformer.rst).

## What are transformer pipelines?

Pipelines are objects that consist of a series of steps (transformer objects), with the final step being an estimator.
Pipelines are useful for establishing consistency between the training model and the inference model, capturing all the intermediate steps appropriately.
Furthermore, we can have multiple estimators as steps in pipelines, and when using GridSearch we gain flexibility to for tuning from the hyper-parameters in the one estimator to that of the whole pipeline.

## Custom transformers included in OptimusAI

* `optimus_core.transformers.SelectColumns`: Transfomer to select columns from input data using a list or regex matching string

```python
SelectColumns(regex="imputed_*")
```
* `optimus_core.transformers.DropColumns`: Transfomer to drop columns from input data using a list or regex matching string

```python
DropColumns(items=["col1", "col2"])
```
* `optimus_core.transformers.DropAllNull`: Transfomer to drop columns from input data where all values in the column are null

```python
DropAllNull()
```
* `optimus_core.transformers.NumExprEval`: Transfomer to create columns at runtime using valid [NumExpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#supported-operators) string expressions. Expressions are carried out sequentially.

```python
NumExprEval(exprs=["C=A+B", "E=sin(A)+tan(C)/cos(B)"])
```
* `optimus_core.transformers.SklearnTransform`: That wraps a default Sklearn transformer to be compatible with the OptimusAI training pipeline. Default sklearn transformers tend to return transformed results as `np.ndarray`. This custom Transformer ensures that DataFrames are returned. It will retain the original column names.

```python
SklearnTransform(transformer=SimpleImputer())
```
* `optimus_core.transformers.MissForestImpute`: Transfomer that uses a [Random Forest Based Imputation method](https://github.com/epsilon-machine/missingpy)

```python
MissForestImpute(max_iter=5, n_estimators=30)
```
## How to create custom transformer

1. Place your custom transformer in `optimus_core/transformers`, and inherit from `optimus_core.transformers.Transformer`.
2. Implement a `.fit` function for this transformer that learns the state of the data to be processed. Ensure you call the `.check_x()` function checking if the passed `x` is a Pandas `DataFrame`.
3. Implement the `.transform` function. This function does the transformation/data manipulation of the original DataFrame. Similarly call the `.check_x()` function at the beginning of the routine. You should be returning a Pandas `DataFrame`.
