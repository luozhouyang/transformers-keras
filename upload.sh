#!/usr/bin/env bash

source activate datasets
python setup.py sdist bdist_wheel
twine upload dist/*

rm -rf build
rm -rf dist
rm -rf *.egg-info
