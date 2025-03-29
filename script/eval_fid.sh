#!/bin/bash

### Calculating FID (plausibility) ###

### START USAGE ###
# sh script/eval_fid.sh ${EXPID} ${EPOCH} 
### END USAGE ###

EXPID=$1
EPOCH=$2


PROJECT_ROOT="$(pwd)"

GT_PATH="$PROJECT_ROOT/dataset/OPA/com_pic_testpos299"
EVAL_PATH="$PROJECT_ROOT/result/${EXPID}/eval/${EPOCH}/images299/"

python eval/fid_resize299.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
python eval/fid_score.py ${EVAL_PATH} ${GT_PATH} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
