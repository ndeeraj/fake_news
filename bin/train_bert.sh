#!/bin/bash

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd ${DIR}/../src

PYTHONPATH=. \
python fake_news/bert/train.py

popd