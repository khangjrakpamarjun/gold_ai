#!/usr/bin/env python
import subprocess
from datetime import datetime, timedelta

import yaml

# Specify the start timestamp
start_ts = datetime.now()
end_ts = start_ts - timedelta(weeks=12)

# Specify the time delta
delta = timedelta(hours=4)  # Example time delta of 1 hour

conf = "conf/base/globals.yml"

params = yaml.safe_load(open(conf))

# Print the generated timestamps
while start_ts >= end_ts:
    params["run_end_date"] = f"{start_ts}+00:00"
    start_ts -= delta
    yaml.dump(params, open(conf, "w"))
    print(params["run_end_date"])
    with subprocess.Popen(
        "kedro run --pipeline live_full_pipe --env live",
        stdout=subprocess.PIPE,
        shell=True,
    ) as p:
        p.communicate()
