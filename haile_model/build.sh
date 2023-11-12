#!/bin/sh -e
cd alchemy
#../lint.sh
poetry version ${1:-"patch"}
VERSION=v$(poetry -s version)
cd ../
git commit -a -m "Release $VERSION"
git tag $VERSION
git push origin $VERSION
