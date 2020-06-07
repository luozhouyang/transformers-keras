#!/usr/bin/env bash

source activate transformers-keras
python setup.py sdist bdist_wheel
python3 -m twine upload dist/*

rm -rf build
rm -rf dist
rm -rf *.egg-info
