#!/bin/bash

# verbose
set -x

epochs=1
step=1e-1
wvecDim=25

outfile="models/rntn_wvecDim_${wvecDim}_step_${step}_2.bin"
#outfile="models/rnn_wvecDim_${wvecDim}_step_${step}_2.bin"

echo $outfile
python2 runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --outputDim 42 --wvecDim $wvecDim

