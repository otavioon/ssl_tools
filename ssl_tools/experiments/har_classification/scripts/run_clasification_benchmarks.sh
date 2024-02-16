#!/bin/bash
TRAIN_SCRIPT="train_and_test.sh"
ROOT_DATASETS_DIR="/workspaces/hiaac-m4/ssl_tools/data/standartized_balanced"
RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="classification_benchmarks/${RUN_ID}"
CWD=".."

declare -A MODELSCRIPTS
declare -A MODELPARAMS

# MLP1x64
MODELSCRIPTS["mlp1x64"]="mlp_classifier.py"
MODELPARAMS["mlp1x64"]="--input_size 360 --hidden_size 64 --num_hidden_layers 1 --num_classes 6"

MODELSCRIPTS["mlp1x64_fft"]="mlp_classifier.py"
MODELPARAMS["mlp1x64_fft"]="--input_size 180 --hidden_size 64 --num_hidden_layers 1 --num_classes 6 --transforms fft"

MODELSCRIPTS["mlp1x64_spectrogram"]="mlp_classifier.py"
MODELPARAMS["mlp1x64_spectrogram"]="--input_size 324 --hidden_size 64 --num_hidden_layers 1 --num_classes 6 --transforms spectrogram"

# MLP2x64
MODELSCRIPTS["mlp2x64"]="mlp_classifier.py"
MODELPARAMS["mlp2x64"]="--input_size 360 --hidden_size 64 --num_hidden_layers 2 --num_classes 6"

MODELSCRIPTS["mlp2x64_fft"]="mlp_classifier.py"
MODELPARAMS["mlp2x64_fft"]="--input_size 180 --hidden_size 64 --num_hidden_layers 2 --num_classes 6 --transforms fft"

MODELSCRIPTS["mlp2x64_spectrogram"]="mlp_classifier.py"
MODELPARAMS["mlp2x64_spectrogram"]="--input_size 324 --hidden_size 64 --num_hidden_layers 2 --num_classes 6 --transforms spectrogram"

# MLP2x64
MODELSCRIPTS["mlp3x64"]="mlp_classifier.py"
MODELPARAMS["mlp3x64"]="--input_size 360 --hidden_size 64 --num_hidden_layers 3 --num_classes 6"

MODELSCRIPTS["mlp3x64_fft"]="mlp_classifier.py"
MODELPARAMS["mlp3x64_fft"]="--input_size 180 --hidden_size 64 --num_hidden_layers 3 --num_classes 6 --transforms fft"

MODELSCRIPTS["mlp3x64_spectrogram"]="mlp_classifier.py"
MODELPARAMS["mlp3x64_spectrogram"]="--input_size 324 --hidden_size 64 --num_hidden_layers 3 --num_classes 6 --transforms spectrogram"

# TNCHead
MODELSCRIPTS["tnchead"]="tnc_head_classifier.py"
MODELPARAMS["tnchead"]="--input_size 360 --num_classes 6"

MODELSCRIPTS["tnchead_fft"]="tnc_head_classifier.py"
MODELPARAMS["tnchead_fft"]="--input_size 180 --num_classes 6 --transforms fft"

MODELSCRIPTS["tnchead_spectrogram"]="tnc_head_classifier.py"
MODELPARAMS["tnchead_spectrogram"]="--input_size 324 --num_classes 6 --transforms spectrogram"

# TFCHead
MODELSCRIPTS["tfchead"]="tfc_head_classifier.py"
MODELPARAMS["tfchead"]="--input_size 360 --num_classes 6"

MODELSCRIPTS["tfchead_fft"]="tfc_head_classifier.py"
MODELPARAMS["tfchead_fft"]="--input_size 180 --num_classes 6 --transforms fft"

MODELSCRIPTS["tfchead_spectrogram"]="tfc_head_classifier.py"
MODELPARAMS["tfchead_spectrogram"]="--input_size 324 --num_classes 6 --transforms spectrogram"

# Simple1DConv
MODELSCRIPTS["simple1Dconv"]="simple1Dconv_classifier.py"
MODELPARAMS["simple1Dconv"]="--num_classes 6 --input_shape [6,60]"

MODELSCRIPTS["simple1Dconv_fft"]="simple1Dconv_classifier.py"
MODELPARAMS["simple1Dconv_fft"]="--num_classes 6 --input_shape [6,30] --transforms fft"

# # Simple2DConv
MODELSCRIPTS["simple2Dconv"]="simple2Dconv_classifier.py"
MODELPARAMS["simple2Dconv"]="--num_classes 6 --input_shape [6,1,60]"

MODELSCRIPTS["simple2Dconv_fft"]="simple2Dconv_classifier.py"
MODELPARAMS["simple2Dconv_fft"]="--num_classes 6 --input_shape [6,1,30] --transforms fft"

# # GRUEncoder
MODELSCRIPTS["gru1l128"]="gru_encoder.py"
MODELPARAMS["gru1l128"]="--num_classes 6 --num_layers 1 --encoding_size 128"

MODELSCRIPTS["gru2l128"]="gru_encoder.py"
MODELPARAMS["gru2l128"]="--num_classes 6 --num_layers 2 --encoding_size 128"

# Run the script

# Iterate over all experiments and call the train_and_test.sh script
# for each experiment and each dataset
for experiment_name in "${!MODELSCRIPTS[@]}"; do
    script=${MODELSCRIPTS[$experiment_name]}
    args=${MODELPARAMS[$experiment_name]}

    for dataset in $ROOT_DATASETS_DIR/*; do
        dataset_name=$(basename $dataset)
        ./train_and_test.sh \
            --log_dir $LOG_DIR \
            --root_dataset_dir $ROOT_DATASETS_DIR \
            --train_dataset $dataset_name \
            --name "${experiment_name}" \
            --script $script \
            --cwd $CWD \
            "${args}"
    done
done
