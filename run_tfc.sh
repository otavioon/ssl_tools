#!/bin/bash

# This script runs the TFC tool using lightining

EXEC_ID="exec_$(date +%Y%m%d_%H%M%S)"

rm -rf checkpoints/tfc_pretrain_c
rm -rf checkpoints/tfc_classifier_c

echo "Pretraining TFC"
python tfc_2_pretrain.py 2>&1 | tee -a ${EXEC_ID}_pretrain.log

echo "Training TFC"
python tfc_2_finetune.py 2>&1 | tee -a ${EXEC_ID}_finetune.log