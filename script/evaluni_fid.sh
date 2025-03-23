#!/bin/bash

### Calculating FID (plausibility) ###

### START USAGE ###
# sh script/eval_fid.sh ${EXPID} ${EPOCH} 
### END USAGE ###

EXPID=$1
EPOCH=$2

python eval/fid_resize299.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "evaluni"

python eval/fid_score.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "evaluni"
