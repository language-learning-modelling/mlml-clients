#!/bin/bash
set -e
source /.env
REQUIRED_VARIABLES=("GITPASS")
echo $GITPASS
for VARIABLE_NAME in "${REQUIRED_VARIABLES[@]}"; do
  value=${!VARIABLE_NAME}
  if [ -n "$value" ]; then
    :
  else
    echo "missing variable :" $VARIABLE_NAME
    exit
  fi
done

#
USER=user
INFRA_FOLDER=/home/training-infrastructure
PERSONAL_NOTES_FOLDER=$INFRA_FOLDER/personal-notes
DATASETS_FOLDER=$INFRA_FOLDER/ll-datasets
TRAINING_LIB_FOLDER=$INFRA_FOLDER/mlm-pipeline
CLIENT_FOLDER=$INFRA_FOLDER/mlml-clients
FOLDERS=($PERSONAL_NOTES_FOLDER $DATASETS_FOLDER $TRAINING_LIB_FOLDER $CLIENT_FOLDER)
PYFOLDERS=($DATASETS_FOLDER $TRAINING_LIB_FOLDER $CLIENT_FOLDER)
REPOS=(https://berstearns:$GITPASS@github.com/berstearns/personal-notes.git
  https://www.github.com/language-learning-modelling/ll-datasets.git
  https://www.github.com/language-learning-modelling/mlm-pipeline.git
  https://www.github.com/language-learning-modelling/mlml-clients.git)
mkdir -p $INFRA_FOLDER
python3 -m venv /venv
/venv/bin/pip install python-dotenv

length=${#FOLDERS[@]}
for ((i = 0; i < $length; i++)); do
  rm -rf ${FOLDERS[$i]}
  git clone --single-branch --depth 1 ${REPOS[$i]} ${FOLDERS[$i]}
done
length=${#PYFOLDERS[@]}
for ((i = 0; i < $length; i++)); do
  FOLDER=${PYFOLDERS[$i]}
  echo $FOLDER
  if [ -f "$FOLDER/requirements.txt" ]; then
    echo "Found requirements.txt in $dir"
    /venv/bin/pip install --ignore-requires-python -r "$FOLDER/requirements.txt"

  # Check if the directory contains a pyproject.toml file
  elif [ -f "$FOLDER/pyproject.toml" ]; then
    echo "Found pyproject.toml in $dir"
    /venv/bin/pip install --ignore-requires-python "$FOLDER"
  fi
done
#cd pip install /home/sagemaker-user/training-infrastructure/mlm-pipeline/
#cp /home/sagemaker-user/training-infrastructure/personal-notes/env-ll-datasets /home/sagemaker-user/training-infrastructure/ll-datasets/clients/.env
#sed -i -e 's/3.12/3/g' /home/sagemaker-user/training-infrastructure/ll-datasets/pyproject.toml
#pip install /home/sagemaker-user/training-infrastructure/ll-datasets/ --no-dependencies
#python3 /home/sagemaker-user/training-infrastructure/ll-datasets/clients/client.py
#cd /home/sagemaker-user/training-infrastructure/mlml-clients/newclient/
#ln -s /home/sagemaker-user/training-infrastructure/ll-datasets/clients/outputs/ /home/sagemaker-user/training-infrastructure/mlml-clients/newclient/
#python3 /home/sagemaker-user/training-infrastructure/mlml-clients/newclient/client.py /home/sagemaker-user/training-infrastructure/mlml-clients/newclient/run_configs/train_bert_on_cleaned_efcamdat_all.json
