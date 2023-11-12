#!/bin/sh -e
pip install black==23.3.0
black alchemy/src/. --line-length 88 -v
pip install isort~=5.0
isort alchemy/src/.
