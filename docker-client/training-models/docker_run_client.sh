#!/bin/bash
# alias d=./docker_run_client.sh;d `jo -p base_model_name="bert-base-uncased" run_hash="2024-09-14-20-11" training_checkpoint="checkpoint-50" dataset_name="efcamdat" split="train" training_strategy="FULL+LLM-TOKENIZE"`
# Ideally it will choose between running docker or simply running as a bahs script



# Check if any arguments are provided
if [ "$#" -eq 0 ]; then
    # No arguments, read from stdin
    INPUT=$(cat)
else
    # Arguments are provided, you can access them via $@
    INPUT="$@"
fi


echo $INPUT
INFRASTRUCTURE="docker"
LOCAL_DATASET="local" 
LOCAL_MODEL="local"   
BATCHES="full" 
# LOCAL_DATASET="download-rclone-gdrive" 
# LOCAL_MODEL="download-rclone-gdrive"   
# BATCHES="full" 

# json='{
#   "model": "bert-base-uncased",
#   "dataset": {
#     "name": "efcamdat",
#     "version": "1.0"
#   }
# }'
# Use jq to parse and iterate over the JSON object
# Extract top-level keys and values
# for key in $(echo "$json" | jq -r 'keys[]'); do
#     value=$(echo "$json" | jq -r --arg k "$key" '.[$k]')
#     if echo "$value" | jq -e . >/dev/null 2>&1; then
#         # Value is a JSON object, handle it accordingly
#         echo "$key is a JSON object:"
#         echo "$value" | jq
#     else
#         # Value is a simple value
#         echo "$key: $value"
#     fi
# done

# CONFIG=`jo -p model="bert-base-uncased" dataset="efcamdat"`;
mkdir -p ./run_configs/
echo $INPUT > ./run_configs/`date +%Y-%m-%d-%H:%M`.json



# Combine the values of the variables into a single string for pattern matching
case "$INFRASTRUCTURE+$LOCAL_DATASET+$LOCAL_MODEL+$BATCHES" in

    "docker+local+local+full")
        echo "Processing with local dataset, local model, and training full dataset using docker."
	echo "expects a datasets/ and models/ folder"
	sudo docker run  --gpus all\
		 -e SKIPDATASETDOWNLOAD=true\
		 -e SKIPMODELDOWNLOAD=true\
		 -v ./datasets:/home/training-infrastructure/mlml-clients/newclient/datasets\
		 -v ./models:/home/training-infrastructure/mlml-clients/newclient/models\
		 -it mlm-pipe-ubuntu24.04-cuda12.04:client $@
        ;;

    "docker+download-rclone-gdrive+download-rclone-gdrive+full")
        echo "Processing with rclone dataset and model from gdrive, training full dataset."
	echo "expects a datasets/ and models/ folder"
	sudo docker run  --gpus all\
		 -e SKIPDATASETDOWNLOAD=false\
		 -e SKIPMODELDOWNLOAD=false\
		 -v ./datasets:/home/training-infrastructure/mlml-clients/newclient/datasets\
		 -v ./models:/home/training-infrastructure/mlml-clients/newclient/models\
		 -it mlm-pipe-ubuntu24.04-cuda12.04:client $@
        # Add commands specific to this combination
        ;;

    *)
        echo "Invalid configuration combination: LOCAL_DATASET=$LOCAL_DATASET, LOCAL_MODEL=$LOCAL_MODEL, BATCHES=$BATCHES"
        # Add error handling for invalid configurations
        ;;
esac

