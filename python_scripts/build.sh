#!/bin/bash

pip uninstall ganslate -y
pushd ganslate
python setup.py sdist bdist_wheel
pip install dist/ganslate-0.1.1-py3-none-any.whl

echo "installation done!"
