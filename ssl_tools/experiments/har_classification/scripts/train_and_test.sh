#!/bin/bash

# Default parameters
ACCELERATOR="cpu"
DEVICES=1
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=256
EPOCHS=300
LEARNING_RATE=0.001
PATIENCE=30
WORKERS=8

# Print usage message
usage() {
    echo "This script runs the specified script on train dataset and then tests on all other datasets in the root dataset directory."
    echo ""

    echo "Usage: $0 --log_dir <LOG_DIR> --root_dataset_dir <ROOT_DATA_DIR> --train_dataset <TRAIN_DATASET> --script <SCRIPT> --name <NAME> [--cwd <CURRENT_WORKING_DIRECTORY>] <SCRIPT_ARGS>"
    echo "  --log_dir: Path to the directory where the logs will be saved"
    echo "  --root_dataset_dir: Path to the root directory containing the datasets"
    echo "  --train_dataset: Name of the dataset to use for training (e.g. KuHar)"
    echo "  --script: Path to the script to run"
    echo "  --name: Name of the experiment"
    echo "  --cwd: Path to the current working directory (default: .)"
    echo "  <SCRIPT_ARGS>: Additional arguments to pass to the script"

    echo ""

    echo "Example:"
    echo "  $0 --log_dir logs --root_dataset_dir datasets --train_dataset KuHar --script tnc_head_classifier.py --name tnc_head --cwd .. --input_size 180 --num_classes 6 --transforms fft"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --log_dir)
        LOG_DIR="$2"
        shift 2
        ;;
    --root_dataset_dir)
        ROOT_DATA_DIR="$2"
        shift 2
        ;;
    --train_dataset)
        TRAIN_DATASET="$2"
        shift 2
        ;;
    --script)
        SCRIPT="$2"
        shift 2
        ;;
    --cwd)
        CWD="$2"
        shift 2
        ;;
    --name)
        NAME="$2"
        shift 2
        ;;
    *)
        SCRIPT_ARGS="$@"
        break
        ;;
    esac
done

# Check if required parameters are provided
if [[ -z $LOG_DIR || -z $ROOT_DATA_DIR || -z $TRAIN_DATASET || -z $SCRIPT || -z $NAME ]]; then
    usage
    exit 1
fi

################################################################################

# Run the training script
write_and_run_fit() {
    local script_file="$1"
    local output_dir="$2"
    local config_file="$output_dir/config.yaml"

    # Create output directory
    mkdir -p "$output_dir"

    # Write the config file
    echo "Writing config file...."
    python "$SCRIPT" fit \
        --print_config \
        --data "$TRAIN_DATASET" \
        --log_dir "$LOG_DIR" \
        --stage train \
        --name "$NAME" \
        --run_id "${RUN_ID}" \
        --accelerator "$ACCELERATOR" \
        --devices "$DEVICES" \
        --batch_size "$TRAIN_BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --checkpoint_metric "val_loss" \
        --checkpoint_metric_mode "min" \
        --num_workers $WORKERS \
        --patience "$PATIENCE" \
        ${SCRIPT_ARGS} >"$(realpath "$config_file")"

    return_code=$?
    # Check if the script execution failed
    if [ $return_code -ne 0 ]; then
        echo "Train script execution failed with return code: $return_code"
        return $return_code
    fi

    echo "Config file saved to $(realpath "$config_file")"
    # Run the script!
    echo "Running train experiment from config file..."
    python "$SCRIPT" fit \
        --config "$config_file" \
        > >(tee "$output_dir/stdout.log") \
        2> >(tee "$output_dir/stderr.log" >&2)
    return_code=$?

    return $return_code
}

# # Run the testing script
write_and_run_test() {
    local script_file="$1"
    local output_dir="$2"
    local target_dataset="$3"
    local checkpoint_path="$LOG_DIR/train/$NAME/$RUN_ID/checkpoints/last.ckpt"
    local config_file="$output_dir/config.yaml"

    mkdir -p "$output_dir"

    # Run the script
    echo "Writing config file...."
    python "$SCRIPT" test \
        --print_config \
        --data "$ROOT_DATA_DIR/$target_dataset" \
        --log_dir "$LOG_DIR" \
        --stage test \
        --name "$NAME" \
        --run_id "$(basename "$output_dir")" \
        --accelerator "$ACCELERATOR" \
        --devices "$DEVICES" \
        --batch_size "$TEST_BATCH_SIZE" \
        --load $checkpoint_path \
        ${SCRIPT_ARGS} >"$config_file"

    return_code=$?

    if [ $return_code -ne 0 ]; then
        echo "Test script execution failed with return code: $return_code"
        return $return_code
    fi

    echo "Config file saved to $(realpath "$config_file")"

    # Run the script
    echo "Running test experiment from config file ($target_dataset)..."
    python "$SCRIPT" test \
        --config "$config_file" \
        > >(tee "$output_dir/stdout.log") \
        2> >(tee "$output_dir/stderr.log" >&2)
    return_code=$?
    return $return_code
}

################################################################################

# Set default value for current working directory if not provided
CWD="${CWD:-.}"
CWD=$(realpath "$CWD")
# Resolve paths
ROOT_DATA_DIR=$(realpath "$ROOT_DATA_DIR")
TRAIN_DATASET="${ROOT_DATA_DIR}/${TRAIN_DATASET}"
RUN_ID=$(basename "$TRAIN_DATASET")

# Output collected parameters
echo "************************* Experiment Parameters *************************"
echo "Name: $NAME"
echo "Log Directory: $LOG_DIR"
echo "Root Dataset Directory: $ROOT_DATA_DIR"
echo "Train Dataset: $TRAIN_DATASET"
echo "Script: $SCRIPT"
echo "Current Working Directory: $CWD"
echo "Script Args: ${SCRIPT_ARGS[@]}"
echo "Run ID: $RUN_ID"
echo "-------------------------------------------------------------------------"

# Change directory to the specified current working directory
cd "$CWD" || exit 1

# ---- TRAINING ----
# Create the train output directory
TRAIN_OUTPUT_DIR="$LOG_DIR/train/$NAME/$RUN_ID/"

# Run the training script
write_and_run_fit "$SCRIPT" "$TRAIN_OUTPUT_DIR"
echo "-------------------------------------------------------------------------"

return_code=$?
if [ $return_code -ne 0 ]; then
    echo "Train script execution failed with return code: $return_code"
    echo "Exiting..."
    echo "*************************************************************************"
    echo ""
    exit $return_code
fi

# ---- TESTING ----
# Run the testing script for each dataset
for dataset_path in "$ROOT_DATA_DIR"/*; do
    dataset=$(basename "$dataset_path")
    echo " ---------------------- Testing on $dataset... ----------------------"
    TEST_OUTPUT_DIR="$LOG_DIR/test/$NAME/train_on-${RUN_ID}-test_on-$dataset/"
    write_and_run_test "$SCRIPT" "$TEST_OUTPUT_DIR" "$dataset"
    echo "-------------------------------------------------------------------------"
done

echo "All done!"
echo "*************************************************************************"
exit 0
