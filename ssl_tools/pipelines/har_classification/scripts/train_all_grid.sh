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

echo "Entering to $CWD"
cd $CWD

for config_name in "${!MODELCONFIGS[@]}"; 
do
    script=${MODELCONFIGS[$config_name]}
    base_config="${BASE_CONFIG_DIR}/${config_name}.yaml"
    for dataset in $(ls $ROOT_DATA_DIR);
    do
        echo " ************ Training $config_name on $dataset ************"
        python $script train --config $base_config \
            --data $ROOT_DATA_DIR/$dataset \
            --experiment_name $EXPERIMENT_NAME \
            --accelerator $ACCELERATOR \
            --devices $DEVICES \
            --max_epochs $EPOCHS \
            --patience $PATIENCE \
            --log_dir $LOG_DIR \
            --checkpoint_monitor_metric $CHECKPOINT_METRIC \
            --checkpoint_monitor_mode "min" \
            --model_name $config_name \
            --model_tags "{'model': '${config_name}', 'trained_on': '${dataset}', 'stage': 'train'}"
    done
done
