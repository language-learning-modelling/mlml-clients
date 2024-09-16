#!/bin/bash
git pull origin;git -C ../../mlm-pipeline pull origin;/venv/bin/pip install ../../mlm-pipeline;/venv/bin/pip install accelerate -U;/venv/bin/python train_client.py `jo -p base_model_name="bert-base-uncased" run_hash="2024-09-14-20-11" training_checkpoint="checkpoint-270" dataset_name="efcamdat" split="train" training_strategy="FULL+LLM-TOKENIZE"`
# run_hash="2024-09-14-20-11" training_checkpoint="checkpoint-50"
