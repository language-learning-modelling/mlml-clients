#!/bin/bash
source ./newenv/bin/activate && pip install ../../mlm-pipeline && python3 -i -W ignore predict.py run_configs/c4200m_predict_efcamdat_test_data.json
# pip install ../../mlm-pipeline;python3 -W ignore predict.py run_configs/c4200m_predict_efcamdat_test_data.json
