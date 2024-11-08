#!/bin/bash
# var for session name (to avoid repeated occurences)
PYTHONBIN="/home/berstearns/projects/language-learning-modelling/mlml-clients/newclient/newenv/bin/python3"
SCRIPTFP="predict.py"
MAX_NUM_TO_PROCESS=80
# sn=xyz
DATASPLITS=()
BATCH_SIZE=20
TOP_K=100

# models: bert-base-uncased, mosaic-bert-base
##################
## CELVA FULL   ##
##################
SPLIT=""
DATASET="CELVA"
INPUT_BATCH_FOLDER="./datasets/${DATASET}/tokenization_batch"
OUTPUT_BATCH_FOLDER="./datasets/${DATASET}/predictions_batch"
MODEL_NAME="bert-base-uncased"
MODEL_CHECKPOINT="./models/${MODEL_NAME}"
##################
## EFCAMDAT TRAIN/TEST##
##################
#SPLIT="test"
#DATASET="EFCAMDAT"
#MODEL_NAME="bert-base-uncased"
#MODEL_CHECKPOINT="./models/${MODEL_NAME}"
#INPUT_BATCH_FOLDER="./datasets/${DATASET}/tokenization_batch/${SPLIT}"
#OUTPUT_BATCH_FOLDER="./datasets/${DATASET}/predictions_batch/${SPLIT}"
#FINALIZED_BATCH_FOLDER="./datasets/${DATASET}/finalized/${MODEL_NAME}"

for INPUTFILENAME in $(ls $INPUT_BATCH_FOLDER -p | grep -v /); do
  OUTPUTFILENAME=${INPUTFILENAME}_${MODEL_NAME}
  echo $OUTPUTFILENAME
  EXPECTED_JSON_OUTPUT=${OUTPUT_BATCH_FOLDER}/$OUTPUTFILENAME.json
  EXPECTED_JSON_GZIP_OUTPUT=${OUTPUT_BATCH_FOLDER}/$OUTPUTFILENAME.json.gz
  EXPECTED_JSON_COMPACT_OUTPUT=${OUTPUT_BATCH_FOLDER}/$OUTPUTFILENAME.json.compact
  #TEST=`wc -l $EXPECTED_OUTPUT 2> /dev/null | awk -F ' ' '{ print $1 }' `
  #if [ -n "$TEST" ] && [ "$TEST" -gt 0 ]
  TEST=$(ls $OUTPUT_BATCH_FOLDER | grep -e $OUTPUTFILENAME)
  TEST2=$(ls $FINALIZED_BATCH_FOLDER | grep -e $OUTPUTFILENAME)
  #echo $INPUTFILENAME;
  #echo $TEST;
  #read -p name
  NOTEST=""
  if [[ "$NOTEST" == "true" ]]; then
    TEST=""
    TEST2=""
  fi
  if [ -n "$TEST" ]; then
    :
  elif [ -n "$TEST2" ]; then
    :
  else
    if [ "${#DATASPLITS[@]}" -lt $MAX_NUM_TO_PROCESS ]; then
      if [[ true ]]; then # $INPUTFILENAME == *ag.json*
        echo "$INPUTFILENAME will be processed"
        DATASPLITS+=($INPUTFILENAME)
      fi
    fi
  fi
done
exit
for i in ${!DATASPLITS[@]}; do
  INPUTFILENAME=${DATASPLITS[$i]}
  FILEPATH="${INPUT_BATCH_FOLDER}/${INPUTFILENAME}"
  #CONFIG={"input_fp": "$FILEPATH","output_folder":"$OUTPUT_BATCH_FOLDER"}
  #CONFIG=\''{"input_fp": "'"$FILEPATH"',"output_folder": "'"$OUTPUT_BATCH_FOLDER"'"}'\'
  CONFIG=$(jo -p input_fp=$FILEPATH output_folder=$OUTPUT_BATCH_FOLDER model_checkpoint=$MODEL_CHECKPOINT batch_size=$BATCH_SIZE top_k=$TOP_K)
  COMMAND="${PYTHONBIN} -W ignore ${SCRIPTFP} $CONFIG" # -i
  echo $i "->" ${DATASPLITS[$i]}
  echo $CONFIG
  # echo $CONFIG #tmux new-window -t "$sn:$((i+1))" -n "${INPUTFILENAME:(-3)}" "zsh -c script.py"
  $COMMAND
  EXPECTED_PARTIAL_JSON_OUTPUT="${OUTPUT_BATCH_FOLDER}/partial/$OUTPUTFILENAME.json"
  TEST=$(ls $OUTPUT_BATCH_FOLDER | grep -e $OUTPUTFILENAME)
  if [ -n "$TEST" ]; then
    jq -c . <$EXPECTED_JSON_OUTPUT >$EXPECTED_JSON_OUTPUT'.compact'
    gzip -9 $EXPECTED_JSON_OUTPUT'.compact'
    rm $EXPECTED_PARTIAL_JSON_OUTPUT
  fi
done
