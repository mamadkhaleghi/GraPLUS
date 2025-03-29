#!/bin/bash

### Calculating Variety LPIPS (diversity) ###

### START USAGE ###
# sh script/eval_lpips.sh ${EXPID} ${EPOCH}
### END USAGE ###

EXPID=$1
EPOCH=$2
PROJECT_ROOT="$(pwd)"
EVALUNI_IMAGES="$PROJECT_ROOT/result/${EXPID}/evaluni/${EPOCH}/images/"

python eval/lpips_1dir.py -d ${EVALUNI_IMAGES} --expid ${EXPID} --epoch ${EPOCH} --eval_type "evaluni" --repeat 10 --use_gpu
