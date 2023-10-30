#!/bin/bash

# # This script runs the TFC tool using lightining

# EXEC_ID="exec_$(date +%Y%m%d_%H%M%S)"

# rm -rf checkpoints/tfc_pretrain_c
# rm -rf checkpoints/tfc_classifier_c

# echo "Pretraining TFC"
# CUDA_VISIBLE_DEVICES=0 python tfc_2_pretrain.py 2>&1 | tee -a ${EXEC_ID}_pretrain.log

# echo "Training TFC"
# CUDA_VISIBLE_DEVICES=0 python tfc_2_finetune.py 2>&1 | tee -a ${EXEC_ID}_finetune.log

# Initialize variables with default values
gpu=""
dataset=""

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --gpu GPU_ID     Specify the GPU ID."
    echo "  --percentage PERCENTAGE   Specify the percentage value."
    echo "  -h, --help       Show this help message."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            echo "ErrorKKK: Unknown option or missing argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check for missing required arguments
if [ -z "$gpu" ] || [ -z "$dataset" ]; then
    echo "E////rror: Missing one or more required arguments."
    show_help
    exit 1
fi

##### Run TF-C ######
# This script runs the TFC tool using lightining

EXEC_ID="exec_${dataset}_$(date +%Y%m%d_%H%M%S)"

rm -rf checkpoints/tfc_pretrain_c
rm -rf checkpoints/tfc_classifier_c

echo "Pretraining TFC"
CUDA_VISIBLE_DEVICES=${gpu} python tfc_2_pretrain.py --dataset ${dataset} 2>&1

echo "Training TFC"
CUDA_VISIBLE_DEVICES=${gpu} python tfc_2_finetune.py --dataset ${dataset} 2>&1