# `pi_connector`

## Overview
`optimus_core.pi_connector` subpackage provides python classes to extract data using
[PI Web API](https://techsupport.osisoft.com/Documentation/PI-Web-API/help.html).

These classes might help you to preprocess or aggregate extracted data:
  - [`OAISummaryStreams`](../../../../../docs/build/apidoc/optimus_core/optimus_core.pi_connector.rst) to summarizes all data captured within time interval (includes total, mean, std dev, last, etc).
  - [`OAIRecordedStreams`](../../../../../docs/build/apidoc/optimus_core/optimus_core.pi_connector.rst) to extract all data changes captured within timeframe specified
  - [`OAICurrentValueStreams`](../../../../../docs/build/apidoc/optimus_core/optimus_core.pi_connector.rst) to extract current available data

Learn more about subpackage structure, function and class interfaces in the [API section](../../../../../docs/build/apidoc/optimus_core/optimus_core.pi_connector.rst).

## User guide

```{eval-rst}
.. toctree::
   :maxdepth: 1

   ../../../docs/pi_connector/data_ingestion.ipynb
 ```

