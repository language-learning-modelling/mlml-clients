#!/bin/bash
git pull origin;git -C ../../mlm-pipeline pull origin;/venv/bin/pip install ../../mlm-pipeline;/venv/bin/pip install accelerate -U;/venv/bin/python train_client.py `jo -p model_checkpoint="bert-base-uncased" dataset_name="efcamdat" split="train" training_strategy="FULL+LLM-TOKENIZE"`
