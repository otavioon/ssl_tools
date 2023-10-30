#!/bin/bash

# Initialize variables with default values
gpu=""

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --gpu GPU_ID     Specify the GPU ID."
    echo "  -h, --help       Show this help message."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            gpu="$2"
            shift 2
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option or missing argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check for missing required arguments
if [ -z "$gpu" ]; then
    echo "Error: Missing one or more required arguments."
    show_help
    exit 1
fi

###### RUN TFC ######

for dataset in UCI KuHar MotionSense RealWorld_thigh RealWorld_waist WISDM; do
    ./single_run.sh --gpu ${gpu} --dataset ${dataset};
done