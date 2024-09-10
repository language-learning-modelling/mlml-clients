#!/bin/bash
source /create_dict_var.sh

if [ -n "$1" ];
then
	for key in "${!dict[@]}"; do
	    echo "$key: ${dict[$key]}"
	done;
	if ! $SKIPDATASETDOWNLOAD;
	then
		cd /home/training-infrastructure/ll-datasets/clients/
		/venv/bin/python ${dict['dataset']}"_client.py"
		ln -s /home/training-infrastructure/ll-datasets/clients/outputs /home/training-infrastructure/mlml-clients/newclient/datasets
	fi 	
	if ! $SKIPMODELDOWNLOAD;
	then
		cd /home/training-infrastructure/mlml-clients/newclient
		/venv/bin/python download_client.py /home/training-infrastructure/mlml-clients/newclient/run_configs/download_bert_base_uncased.json
        fi	
	cd /home/training-infrastructure/mlml-clients/newclient 
	/venv/bin/pip install accelerate -U
	#/venv/bin/python train_client.py `jo -p  model_checkpoint="dict['model']"\
        #					 dataset_name="dict['dataset']"`
        #					 training_strategy="dict['training_strategy']"`
fi
bash
