#!/bin/bash

source ./common_vars.sh

EXPERIMENT_NAME="har_train"
BASE_CONFIG_DIR="${CWD}/configs/base/train"


usage() {
    echo "This script trains a model over several datasets."
    echo ""
    echo "Usage: $0 --mode <window|concatenated> --root_dataset_dir <ROOT_DATA_DIR>"
    echo "  --root_dataset_dir: Path to the root directory containing the datasets"
}

################################################################################
while [[ $# -gt 0 ]]; do
    case $1 in
    --root_dataset_dir)
        ROOT_DATA_DIR="$2"
        shift 2
        ;;
    --mode)
        MODE="$2"
        shift 2
        ;;

    *)
        usage
        exit 1
        ;;
    esac
done


if [ -z "$ROOT_DATA_DIR" ] ; then
    usage
    exit 1
fi

if [ "$MODE" = "window" ]; then
    source configs_train_windowed_view.sh
elif [ "$MODE" = "concatenated" ]; then
    source configs_train_concatenated_view.sh
else
    usage
    exit 1
fi

################################################################################
# Get a list of subdirectories
subdirs=("$ROOT_DATA_DIR"/*)
# Calculate the total number of iterations
total_iterations=$(( ${#subdirs[@]} - 1 ))


echo "Entering to $CWD"
cd $CWD

for config_name in "${!MODELCONFIGS[@]}"; 
do
    script=${MODELCONFIGS[$config_name]}
    base_config="${BASE_CONFIG_DIR}/${config_name}.yaml"
    for (( iteration = 0; iteration < total_iterations; iteration++ ));
    do
        X=()

        # Populate X and Y arrays
        for (( i = 0; i < total_iterations; i++ )); do
            if (( i != iteration )); then
                X+=("${subdirs[i]}")
            fi
        done

        x_json=$(printf '%s\n' "${X[@]}" | jq -R . | jq -s -c . | tr -d '\n')
        x_json=${x_json//\"/\'}
        x_basenames=$(basename -a "${X[@]}" | tr '\n' '+')
        x_basenames=${x_basenames%+} # Remove the trailing '+'

        echo " ************ Training $config_name on $x_basenames ************"
        python $script train --config $base_config \
            --num_classes 7 \
            --data "${x_json}" \
            --experiment_name $EXPERIMENT_NAME \
            --accelerator $ACCELERATOR \
            --devices $DEVICES \
            --max_epochs $EPOCHS \
            --patience $PATIENCE \
            --log_dir $LOG_DIR \
            --checkpoint_monitor_metric $CHECKPOINT_METRIC \
            --checkpoint_monitor_mode "min" \
            --model_name $config_name \
            --model_tags "{'model': '${config_name}', 'trained_on': '${x_basenames}', 'stage': 'train'}"
    done
done
