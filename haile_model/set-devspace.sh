#!/bin/bash -e
export USER=$1

if ! command -v yq &> /dev/null
then
	echo "yq is not found, attempting to install..."
	brew install yq
fi

if [ -z "$USER" ]; then
	echo "Usage: $0 username"
	exit
fi

export LOCALNAME="console-$USER"

cp etc/devspace.yaml ./
yq '.dev.console.labelSelector.app = env(LOCALNAME)' -i devspace.yaml

cp etc/deployment.yaml ./
yq '.metadata.name = env(LOCALNAME)' -i deployment.yaml
yq '.spec.selector.matchLabels.app = env(LOCALNAME)' -i deployment.yaml
yq '.spec.template.metadata.labels.app = env(LOCALNAME)' -i deployment.yaml
yq '.spec.template.spec.containers[0].name = env(LOCALNAME)' -i deployment.yaml

echo "base_dir: /data/${USER}" > alchemy/conf/local/globals.yml
devspace use namespace ml
