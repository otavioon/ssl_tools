#!/bin/bash

cd ..

# ----------------- Parameters -----------------

# LOG_DIR will be: anomaly_detection_results/2021-08-25_16-00-00/
LOG_DIR="anomaly_detection_results/$(date +'%Y-%m-%d_%H-%M-%S')"
TRAIN_DATA="data/windowed_16_overlap_0_rate_10min_df_scaled.csv"
TEST_DATA="data/windowed_16_overlap_0_rate_10min_df_scaled.csv"
TRAIN_PARTICIPANT=11
TEST_PARTICIPANT=11
EPOCHS=1
INPUT_SHAPE="[1, 16]"
LEARNING_RATE=0.001
NAME="cae"


# ----------------- Train -----------------
echo "Training CAE Autoencoder"
echo "Log directory: ${LOG_DIR}. Train participant: ${TRAIN_PARTICIPANT}. Test participant: ${TEST_PARTICIPANT}"

# PARTICIPANT = From 0 to 76 (77 participants)
python ./cae.py fit \
    --data  ${TRAIN_DATA} \
    --participant_ids "${TRAIN_PARTICIPANT}" \
    --input_shape "${INPUT_SHAPE}" \
    --epochs "${EPOCHS}" \
    --batch_size 64 \
    --augment true \
    --learning_rate "${LEARNING_RATE}" \
    --log_dir "${LOG_DIR}" \
    --stage train \
    --name "${NAME}" \
    --run_id "${TRAIN_PARTICIPANT}" \
    --accelerator "gpu" \
    --devices 1



# ----------------- Test -----------------

python ./cae.py  test \
    --train_data "${TRAIN_DATA}" \
    --test_data "${TEST_DATA}" \
    --train_participant "${TRAIN_PARTICIPANT}" \
    --test_participant "${TEST_PARTICIPANT}" \
    --input_shape "${INPUT_SHAPE}" \
    --batch_size 256 \
    --load "${LOG_DIR}/train/${NAME}/${TRAIN_PARTICIPANT}/checkpoints/last.ckpt" \
    --log_dir "${LOG_DIR}" \
    --stage test \
    --name "${NAME}" \
    --run_id "${TRAIN_PARTICIPANT}_${TEST_PARTICIPANT}" \
    --accelerator "gpu" \
    --devices 1