#!/bin/bash

# verbose
set -x

testdir=$1
data=test

echo $testdir
python2 runNNet.py --model_directory $testdir --test --data $data
