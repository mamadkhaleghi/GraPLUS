#!/bin/bash

### Calculating Spatial Precision ###

### START USAGE ###
# sh script/eval_spatial_precision.sh ${EXPID} ${EPOCH} 
### END USAGE ###

EXPID=$1
EPOCH=$2

python eval/spatial_precision.py ${EXPID} ${EPOCH}