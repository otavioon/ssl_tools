#!/bin/bash

# set -x
source ./common_vars.sh
source configs_finetune_windowed_view.sh

EXPERIMENT_NAME="har_finetune"
BASE_CONFIG_DIR="${CWD}/configs/base/finetune"

usage() {
    echo "This script trains a model over several datasets."
    echo ""
    echo "Usage: $0 --root_dataset_dir <ROOT_DATA_DIR>"
    echo "  --root_dataset_dir: Path to the root directory containing the datasets"
}

################################################################################
while [[ $# -gt 0 ]]; do
    case $1 in
    --root_dataset_dir)
        ROOT_DATA_DIR="$2"
        shift 2
        ;;
    *)
        usage
        exit 1
        ;;
    esac
done


if [ -z "$ROOT_DATA_DIR" ]; then
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
        for (( i = 0; i < total_iterations; i++ )); 
        do
            if (( i != iteration )); then
                X+=("${subdirs[i]}")
            fi
        done

        source_dataset_json=$(printf '%s\n' "${X[@]}" | jq -R . | jq -s -c . | tr -d '\n')
        source_dataset_json=${source_dataset_json//\"/\'}
        source_dataset_basenames=$(basename -a "${X[@]}" | tr '\n' '+')
        source_dataset_basenames=${source_dataset_basenames%+} # Remove the trailing '+'
            
        for target_dataset in $(ls $ROOT_DATA_DIR);
        do
            echo " ************ Training $config_name on $target_dataset using model trained on ${source_dataset_basenames} ************"
            
            model_name="${config_name}"
            if [[ $model_name == *"_non_freeze"* ]]; then
                # Remove "_non_freeze" from the model_name
                model_name="${model_name//_non_freeze/}"
            fi  

            python $script finetune --config $base_config \
                --num_classes 7 \
                --data $ROOT_DATA_DIR/$target_dataset \
                --experiment_name $EXPERIMENT_NAME \
                --accelerator $ACCELERATOR \
                --devices $DEVICES \
                --max_epochs $EPOCHS \
                --patience $PATIENCE \
                --log_dir $LOG_DIR \
                --checkpoint_monitor_metric $CHECKPOINT_METRIC \
                --checkpoint_monitor_mode "min" \
                --model_name $config_name \
                --model_tags "{'model': '${config_name}', 'trained_on': '${source_dataset_basenames}', 'finetune_on': '${target_dataset}', 'stage': 'finetune'}" \
                --registered_model_name $model_name \
                --registered_model_tags "{'trained_on': '${source_dataset_basenames}', 'stage': 'train'}" 
        done
    done
done
