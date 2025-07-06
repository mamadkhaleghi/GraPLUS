#!/bin/bash

### Calculating Spatial Precision (New Approach) ###

### START USAGE ###
# sh script/eval_spatial_precision_new.sh ${EXPID} ${EPOCH} 
### END USAGE ###

EXPID=$1
EPOCH=$2

python eval/spatial_precision_new.py ${EXPID} ${EPOCH}
