# COVID Anomaly Detection and Binary Classification

This folder contains the code to train and evaluate anomaly detection and binary classification models for COVID-19 detection. It has several experiments to train and evaluate different models.

## Executing experiments

Examples of how to execute the experiments are in the `scripts` folder.

## Results Directory Structure

The results are saved in the `anomaly_detection_results` directory. The directory structure is as follows:


```
# Ouput directory structure
# anomaly_detection_results/
# ├── 2021-08-25_16-00-00
# │   ├── train
# │   │   ├── lstm_ae
# │   │   │   ├── 0
# │   │   │   │   ├── checkpoints
# │   │   │   │   │   ├── last.ckpt
# │   │   │   │   ├── metrics.csv
# │   │   │   │   ├── hparams.yaml
# │   ├── test
# │   │   ├── lstm_ae
# │   │   │   ├── 0_0
# │   │   │   │   ├── checkpoints
# │   │   │   │   │   ├── last.ckpt
# │   │   │   │   ├── metrics.csv
# │   │   │   │   ├── hparams.yaml
# │   │   │   │   ├── results.csv

```

## Experiments

- Anomaly detection with LSTM Autoencoder, Convolutional Autoencoder 1D and 2D, Contrastive Convolutional Autoencoder 1D and 2D, and VAE. Yu can opt to use augmentations or not. It will evaluate several metrics such as AUC, F1, Precision, Recall, and using several thresholds functions and losses functions.
- Binary classification with LSTM, CNN, and ResNet. You can opt to use augmentations or not (SMOTE). 