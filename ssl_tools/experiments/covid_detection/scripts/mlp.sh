#!/bin/bash

cd ..

# ----------------- Parameters -----------------

# LOG_DIR will be: anomaly_detection_results/2021-08-25_16-00-00/
LOG_DIR="binary_classification_results/$(date +'%Y-%m-%d_%H-%M-%S')"
TRAIN_DATA="/workspaces/torch/project/aimh/data/windowed_16_overlap_0_rate_10min_df_scaled_train_fold_0.csv"
TEST_DATA_PREFIX="/workspaces/torch/project/aimh/data/windowed_16_overlap_0_rate_10min_df_scaled_test_fold"
TRAIN_FOLD="fold0"
EPOCHS=1
INPUT_SIZE=16
HIDDEN_SIZE=128
NUM_LAYERS=1
LEARNING_RATE=0.001
NAME="mlp"
BALANCE=true


# ----------------- Train -----------------
echo "Training MLP"
echo "Log directory: ${LOG_DIR}. Train data: ${TRAIN_DATA}. Test data: ${TEST_DATA}"

python ./mlp.py fit \
    --data  "${TRAIN_DATA}" \
    --epochs "${EPOCHS}" \
    --batch_size 256 \
    --learning_rate "${LEARNING_RATE}" \
    --input_size "${INPUT_SIZE}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_hidden_layers "${NUM_LAYERS}" \
    --log_dir "${LOG_DIR}" \
    --stage train \
    --name "${NAME}" \
    --run_id "${TRAIN_FOLD}" \
    --accelerator "gpu" \
    --devices 1 \
    --balance ${BALANCE}



# # ----------------- Test -----------------

for i in 0 1 2 3 4; do
    TEST_DATA="${TEST_DATA_PREFIX}_${i}.csv"
    echo "Testing MLP with test data: ${TEST_DATA}"
    python ./mlp.py test \
        --data  "${TEST_DATA}" \
        --batch_size 256 \
        --load "${LOG_DIR}/train/${NAME}/${TRAIN_FOLD}/checkpoints/last.ckpt" \
        --input_size "${INPUT_SIZE}" \
        --hidden_size "${HIDDEN_SIZE}" \
        --num_hidden_layers "${NUM_LAYERS}" \
        --log_dir "${LOG_DIR}" \
        --stage test \
        --name "${NAME}" \
        --run_id "${TRAIN_FOLD}_fold${i}" \
        --accelerator "gpu" \
        --devices 1 
done 