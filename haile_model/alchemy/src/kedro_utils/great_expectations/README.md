# Data validation module
This module helps identify anomalies and outliers in the dataset using [Great Expectations](https://docs.greatexpectations.io/en/latest/intro.html).

You can find an example Jupyter notebook to illustrate data validation with Great Expectations in the `core_pipelines/core_pipelines/kedro_utils/great_expectations/` subfolder of OptimusAI.

## Get started
The following steps will help you set up data validation.

### How to install Kedro Great Expectations

You can install `kedro-great-expectations` using `pip`:

```bash
pip install optimus/packages/kedro_great_expectations-0.3.0-py3-none-any.whl
```

> For McKinsey users, a .whl file for kedro_great_expectations is also available on [box](https://mckinsey.box.com/v/kedro-great-expectations)

#### Use `kedro-great-expectations` in a pipeline

Modify your project's `run.py` to include the following to enable validation on kedro run:

```python
from kedro.context import KedroContext
from kedro_great_expectations import GreatExpectationsMixin

class ProjectContext(GreatExpectationsMixin, KedroContext):
    # refer to sample config in optimus/pipeline/conf/base/kedro_ge.yml
    ge_config_key = "kedro_ge.yml"   # optional, defaults to this value
    ### ...
```
#### Set up and configure Great Expectations

To generate a `kedro_ge.yml` configuration file in the `conf/base` folder, initiate `kedro-great-expectations` as follows:

```bash
kedro ge init
```
The configuration file can be adapted to suit the needs of your project. The class path for custom expectations developed for OptimusAI is included in this file.

> McKinsey users can find more information about configuration on [QB/Protocols](https://one.quantumblack.com/docs/alchemy/kedro_great_expectations/03_user_guide/01_configuration.html).

#### Create a Great Expectations suite

The following commands will help create an empty GE suite for each dataset. Make sure you are in the pipeline folder before executing the commands.

```bash
cd pipeline
kedro ge generate <dataset_name> --empty
kedro ge edit <dataset_name>
```

This will open a Jupyter notebook `dataset_name.ipynb` for editing.

#### Build your Expectations suite

OptimusAI has built some custom expectations that can be used in addition to those provided by GE. These can be found in the `great_expectations_utils.py` file. The custom expectation class and its methods are detailed in the `adtk_custom_expectation.py` file.
Simply copy paste the desired method into the notebook. The example below implements anomaly detection using quantiles.

``` python
from core_pipelines.kedro_utils.great_expectations.great_expectations_utils import *
params = context.params

# Custom Expectation - Quantile anomaly detection
validate_column_quantile_anomaly(batch, params)
```
The parameter file for data validation module is located at `<my_project>/pipeline/conf/base/pipelines/validate/parameters.yml`.

```
dataset_1:
  column_list: ["status_time", "outp_quantity", "inp_quantity", "cu_content", "inp_avg_hardness"]

  data_length:
    min_value: 0
    max_value: 26

  schema:
    "cu_content": "float64"
    "inp_avg_hardness": "float64"
    "inp_quantity": "float64"
    "outp_quantity": "float64"
    "status_time": "object"

  time:
    column: "status_time"
    format: "%Y-%m-%d %H:%M:%S"

  process_window: 8       #  amount of time to complete ops process

  sensor_pair_1:
    first_sensor: "inp_quantity"
    second_sensor: "outp_quantity"

  quantile_anomaly:
    low: 0.01             # Quantile of historical data lower which a value is regarded as anomaly
    high:  0.99           # Quantile of historical data above which a value is regarded as anomaly
```
Parameters are designated by dataset i.e. each data can have their own top level key to differentiate between configurations.


## Custom Expectations

Currently, OptimusAI supports two types of anomaly detection:

- Rule based anomaly detection
- Model based advanced anomaly detection

Both have been implemented using the [anomaly detection toolkit (ADTK) package](https://adtk.readthedocs.io/en/stable/index.html).

### Rule based anomaly detection
The following methods detect anomalies using set rules to detect anomalies:

* Level Shift Anomaly Detection: `create_level_shift_expectation`
This detects level shifts in the dataset by comparing values from two time windows.
* Quantile Anomaly Detection: `validate_column_quantile_anomaly`
This detects anomalies based on quantiles of historical data
* Persist Anomaly Detection: `validate_column_persist_anomaly`
This detects anomalies based on values in a preceding time period.

### Advanced anomaly detection
Sometimes, it is difficult to detect anomalies based on simple rules. Model based anomaly detection can help solve this issue. The following methods are currently available:

* Isolation Forest: `validate_multi_dimension_isolationforest_anomaly`
This method identifies time points as anomalous based isolation forest technique. This is a tree based technique and is highly effective in high dimensional data.
* KMeans Clustering: `validate_multi_dimension_cluster_anomaly`
This method identifies anomalies based on clustering historical data

## FAQ
### Can I add my own expectation?
Yes, you can create your own expectation, as follows:

1. Go to `core_pipelines/core_pipelines/kedro_utils/great_expectations/adtk_custom_expectation.py`
2. Add your function to the class `CustomADTKExpectations`.
3. Include your function in the GE utils file, i.e., `great_expectations_utils.py`.
4. Call this function when creating your GE suite through the Jupyter notebook generated for your dataset.

### What are decorators? How are they used?

Decorators are callable objects that add new functionality to an existing object without modifying its structure. GE provides high-level decorators that help convert our custom functions into a fully-fledged expectation.

We use the `column_aggregate_expectation` decorator from class `MetaPandasDataset`

Refer to the [Great Expectations documentation](https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/index.html#great_expectations.dataset.MetaPandasDataset) for additional options.

### How should I configure my expectation file?

You can configure how validation works for your datasets by using the following [config schema](https://one.quantumblack.com/docs/alchemy/kedro_great_expectations/03_user_guide/01_configuration.html)

### Help! I'm stuck: how can I get in touch?

Please raise an issue [here](https://github.com/McK-Internal/optimus/issues/new/choose). Alternatively, get in touch via slack [#optimus](https://mckinsey-client-cap.slack.com/archives/C9S1RM6SX).


## Example of a Great Expectations validation  notebook
You can find an example Jupyter notebook to illustrate data validation with Great Expectations in the `core_pipelines/core_pipelines/kedro_utils/great_expectations/` subfolder of OptimusAI.

To run the notebook, use the following command in the terminal, within the `pipeline` project directory:

```bash
kedro jupyter notebook
```

Then navigate to the `core/data_validation` subfolder and open the notebook named `example.ipynb`.
