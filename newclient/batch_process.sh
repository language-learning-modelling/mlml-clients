#!/bin/bash
# var for session name (to avoid repeated occurences)
PYTHONBIN="/home/berstearns/projects/language-learning-modelling/mlml-clients/newclient/newenv/bin/python3"
SCRIPTFP="predict.py"
MAX_NUM_TO_PROCESS=2
sn=xyz

# Start the session and window 0 in /etc
#   This will also be the default cwd for new windows created
#   via a binding unless overridden with default-path.
#tmux new-session -s "$sn" -n tokenization -d

# Expect to be in the selva-agreements/clients/poetry-clinet folder 
# Create a bunch of windows, one for each data split
DATASPLITS=()
SPLIT="train"
INPUT_BATCH_FOLDER="./datasets/EFCAMDAT/masked_sentences_batch/${SPLIT}"
OUTPUT_BATCH_FOLDER="./datasets/EFCAMDAT/predictions_batch/${SPLIT}"
MODEL_CHECKPOINT="./models/bert-base-uncased-c4200m-unchaged-vocab-73640000"
BATCH_SIZE=75
TOP_K=10
for FILENAME in `ls $INPUT_BATCH_FOLDER`;
do
	EXPECTED_OUTPUT='${OUTPUT_BATCH_FOLDER}/${FILENAME}'
	TEST=`wc -l $EXPECTED_OUTPUT 2> /dev/null | awk -F ' ' '{ print $1 }' `  
	if [ -n "$TEST" ] && [ "$TEST" -gt 0 ] 
	then
		LINECOUNT=`wc -l $EXPECTED_OUTPUT | awk -F ' ' '{ print $1 }'` 
	else
		if [ "${#DATASPLITS[@]}" -lt $MAX_NUM_TO_PROCESS ]
		then
			echo "$FILENAME will be processed";
			DATASPLITS+=( $FILENAME )
		fi
	fi
done
for i in ${!DATASPLITS[@]}; 
do
    FILENAME=${DATASPLITS[$i]}
    FILEPATH="${INPUT_BATCH_FOLDER}/${FILENAME}"
    #CONFIG={"input_fp": "$FILEPATH","output_folder":"$OUTPUT_BATCH_FOLDER"}
    #CONFIG=\''{"input_fp": "'"$FILEPATH"',"output_folder": "'"$OUTPUT_BATCH_FOLDER"'"}'\'
    CONFIG=`jo -p\
	    input_fp=$FILEPATH\
	    output_folder=$OUTPUT_BATCH_FOLDER\
            model_checkpoint=$MODEL_CHECKPOINT\
            batch_size=$BATCH_SIZE\
            top_k=$TOP_K\
	   `
    COMMAND="${PYTHONBIN} -W ignore ${SCRIPTFP} $CONFIG" 
    echo $i "->" ${DATASPLITS[$i]}
    echo $CONFIG
    # echo $CONFIG
    #tmux new-window -t "$sn:$((i+1))" -n "${FILENAME:(-3)}" "zsh -c script.py"
    $COMMAND &
done

# Set the default cwd for new windows (optional, otherwise defaults to session cwd)
#tmux set-option default-path /

# Select window #1 and attach to the session
#tmux select-window -t "$sn:0"
#tmux -2 attach-session -t "$sn"
