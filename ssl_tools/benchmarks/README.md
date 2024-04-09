# Benchmarks

## Overview

Benchmark A1: Supervised Learning
To run, use: `./main_supervised.py --config configs/benchmarks/benchmark_a1.yaml`

### Models

| **Model**                                                                          	| **Authors**                      	| **Task**       	| **Type**              	| **Input Shape** 	|                       **Python Class**                      	| **Observations**                                                                                                            	|
|------------------------------------------------------------------------------------	|----------------------------------	|----------------	|-----------------------	|:---------------:	|:-----------------------------------------------------------:	|-----------------------------------------------------------------------------------------------------------------------------	|
| [DeepConvLSTM](https://www.mdpi.com/1424-8220/16/1/115)                            	| Ordóñez and Roggen               	| Classification 	| 2D Conv + LSTM        	|    (C, S, T)    	|      ssl_tools.models.nets.deep_conv_lstm.DeepConvLSTM      	|                                                                                                                             	|
| [Simple 1D Convolutional Network](https://www.mdpi.com/1424-8220/16/1/115)         	| Ordóñez and Roggen               	| Classification 	| 1D Conv               	|      (S, T)     	|      ssl_tools.models.nets.convnet.Simple1DConvNetwork      	| 1D Variant of "Baseline CNN", used by Ordóñez and Roggen,  with dropout layers included.                                    	|
| [Simple 2D Convolutional Network](https://www.mdpi.com/1424-8220/16/1/115)         	| Ordóñez and Roggen               	| Classification 	| 2D Conv               	|    (C, S, T)    	|      ssl_tools.models.nets.convnet.Simple2DConvNetwork      	| 2D Variant of "Baseline CNN", used by Ordóñez and Roggen,  with dropout layers included.                                    	|
| [CNN_HaEtAl_1D](https://ieeexplore.ieee.org/document/7379657)                      	| Ha, Yun and Choi                 	| Classification 	| 1D Conv               	|      (S, T)     	|       ssl_tools.models.nets.cnn_ha_etal.CNN_HaEtAl_1D       	| 1D proposed variant.                                                                                                        	|
| [CNN_HaEtAl_2D](https://ieeexplore.ieee.org/document/7379657)                      	| Ha, Yun and Choi                 	| Classification 	| 2D Conv               	|    (C, S, T)    	|       ssl_tools.models.nets.cnn_ha_etal.CNN_HaEtAl_2D       	| 2D proposed variant.                                                                                                        	|
| [CNN PF](https://ieeexplore.ieee.org/document/7727224)                             	| Ha and Choi                      	| Classification 	| 2D Conv               	|    (C, S, T)    	|            ssl_tools.models.nets.cnn_pf.CNN_PF_2D           	| Partial weight sharing in first convolutional layer and  full weight sharing in second convolutional layer.                 	|
| [CNN PPF](https://ieeexplore.ieee.org/document/7727224)                            	| Ha and Choi                      	| Classification 	| 2D Conv               	|    (C, S, T)    	|           ssl_tools.models.nets.cnn_pf.CNN_PFF_2D           	| Partial and full weight sharing in first convolutional layer  and full weight sharing in second convolutional layer.        	|
| [IMU Transformer](https://ieeexplore.ieee.org/document/9393889)                    	| Shavit and Klein                 	| Classification 	| 1D Conv + Transformer 	|      (S, T)     	| ssl_tools.models.nets.imu_transformer.IMUTransformerEncoder 	|                                                                                                                             	|
| [IMU CNN](https://ieeexplore.ieee.org/document/9393889)                            	| Shavit and Klein                 	| Classification 	| 1D Conv               	|      (S, T)     	|         ssl_tools.models.nets.imu_transformer.IMUCNN        	| Baseline CNN for IMUTransnformer work.                                                                                      	|
| [InceptionTime](https://doi.org/10.1007/s10618-020-00710-y)                        	| Fawaz et al.                     	| Classification 	| 1D Conv               	|      (S, T)     	|      ssl_tools.models.nets.inception_time.InceptionTime     	|                                                                                                                             	|
| [1D-ResNet](https://www.mdpi.com/1424-8220/22/8/3094)                              	| Mekruksavanich and Jitpattanakul 	| Classification 	| 1D Conv               	|      (S, T)     	|          ssl_tools.models.nets.resnet_1d.ResNet1D_8         	| Baseline resnet from paper. Uses ELU and 8 residual blocks                                                                  	|
| [1D-ResNet-SE-8](https://www.mdpi.com/1424-8220/22/8/3094)                         	| Mekruksavanich and Jitpattanakul 	| Classification 	| 1D Conv               	|      (S, T)     	|         ssl_tools.models.nets.resnet_1d.ResNetSE1D_8        	| ResNet with Squeeze and Excitation. Uses ELU and 8 residual  blocks                                                         	|
| [1D-ResNet-SE-5](https://ieeexplore.ieee.org/document/9771436)                     	| Mekruksavanich et al.            	| Classification 	| 1D Conv               	|      (S, T)     	|         ssl_tools.models.nets.resnet_1d.ResNetSE1D_5        	| ResNet with Squeeze and Excitation. Uses ReLU and 8 residual  blocks                                                        	|
| [InnoHAR](https://ieeexplore.ieee.org/document/8598871)                            	| Xu et al.                        	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [TCN](https://dl.acm.org/doi/10.1145/3266157.3266221)                              	| Sikder et al.                    	| Classification 	|                       	|                 	|                                                             	| Encoder-Decoder TCN                                                                                                         	|
| [MCNN](https://ieeexplore.ieee.org/document/8975649)                               	| Sikder et al.                    	| Classification 	| 2D Conv               	|   (2, C, S,T)   	| ssl_tools.models.nets.multi_channel_cnn.MultiChannelCNN_HAR 	| First dimension is FFT data and second is Welch Power Density periodgram data. Must adapt dataset to return data like this. 	|
| [HVMAN](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221390) 	| Zhang et al.                     	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [EnsemConvNet](https://link.springer.com/article/10.1007/s11042-020-09537-7)       	| Mukherjee et al.                 	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [LW-CNN](https://ieeexplore.ieee.org/document/9165108)                             	| Tang et al.                      	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [DanHAR](https://www.sciencedirect.com/science/article/pii/S1568494621006499)      	| Gao et al.                       	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [AMH-CNN](https://doi.org/10.1016/j.asoc.2021.107671)                              	| Khan and Ahmad                   	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [SK-CNN](https://doi.org/10.1109/TIM.2021.3102735)                                 	| Zhang et al.                     	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [Light-MHTCN](https://www.sciencedirect.com/science/article/pii/S0957417423006346) 	| Sekaran, Han, and Yin            	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [CNN-HS](https://ieeexplore.ieee.org/document/9744439)                             	| Tang et al.                      	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [DeepSense](https://arxiv.org/pdf/1611.01942.pdf)                                  	| Yao et al.                       	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [LSTM-RNN](https://ieeexplore.ieee.org/document/8843403)                           	| Pienaar and Malekian             	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [Lightweight RNN-LSTM](https://doi.org/10.1016/j.procs.2020.03.289)                	| Agarwal and Alam                 	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [PerRNN](http://dx.doi.org/10.1109/ICC.2019.8761931)                               	| Wang et al.                      	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [DCNN](https://dl.acm.org/doi/10.1145/2733373.2806333)                             	| Jiang and Yin                    	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [GRU-INC](https://www.sciencedirect.com/science/article/abs/pii/S0957417422024381) 	| Mim et al.                       	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [LSTM-CNN](https://ieeexplore.ieee.org/abstract/document/9043535)                  	| Xia et al.                       	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [Resnet-LSTM-CNN](https://link.springer.com/article/10.1007/s13534-020-00160-x)    	| Shojaedini and Beirami           	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [iSPLInception](https://ieeexplore.ieee.org/document/9425494/)                     	| Ronald, Poulose, and Han         	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [CNN-LSTM](https://ieeexplore.ieee.org/document/9065078)                           	| Mutegeki and Han                 	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [BiLSTM](https://ieeexplore.ieee.org/document/9065250)                             	| Thu and Han                      	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [SCbSE-SMFE](https://link.springer.com/article/10.1007/s12652-020-02526-6)         	| Zhang et al.                     	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
| [AttnSense](https://www.ijcai.org/Proceedings/2019/0431.pdf)                       	| Ma et al.                        	| Classification 	|                       	|                 	|                                                             	|                                                                                                                             	|
|                                                                                    	|                                  	|                	|                       	|                 	|                                                             	|                                                                                                                             	|

### Self-Supervised Methods


| **Method** | **Authors** | **Input Shape** | **Python Class** | **Observations** |
|------------|-------------|-----------------|------------------|:----------------:|
| TNC        |             |                 |                  |                  |
| TFC        |             |                 |                  |                  |
| CPC        |             |                 |                  |                  |


### Domain Generalization

| **Method** | **Authors** | **Python Class** | **Observations** |
|------------|-------------|------------------|:----------------:|
| MixUp      |             |                  |                  |
| FixMatch   |             |                  |                  |
| DDG        |             |                  |                  |



## Supervised Benchmarks

Trained and tested in the same dataset (10% - (random, balanced), 20%, 40%, 50%)



| **Model**                       | **KuHar** | **MotionSense** | **RW-Waist** | **RW-Thigh** | **UCI** | **WISDM** |
|---------------------------------|-----------|-----------------|--------------|:------------:|:-------:|-----------|
| DeepConvLSTM                    |           |                 |              |              |         |           |
| Simple 1D Convolutional Network |           |                 |              |              |         |           |
| Simple 2D Convolutional Network |           |                 |              |              |         |           |
| CNN_HaEtAl_1D                   |           |                 |              |              |         |           |
| CNN_HaEtAl_2D                   |           |                 |              |              |         |           |
| CNN PF                          |           |                 |              |              |         |           |
| CNN PPF                         |           |                 |              |              |         |           |
| IMU Transformer                 |           |                 |              |              |         |           |
| IMU CNN                         |           |                 |              |              |         |           |
| InceptionTime                   |           |                 |              |              |         |           |
| 1D-ResNet                       |           |                 |              |              |         |           |
| 1D-ResNet-SE-8                  |           |                 |              |              |         |           |
| 1D-ResNet-SE-5                  |           |                 |              |              |         |           |
| InnoHAR                         |           |                 |              |              |         |           |
| TCN                             |           |                 |              |              |         |           |
| MCNN                            |           |                 |              |              |         |           |
| HVMAN                           |           |                 |              |              |         |           |
| EnsemConvNet                    |           |                 |              |              |         |           |
| LW-CNN                          |           |                 |              |              |         |           |
| DanHAR                          |           |                 |              |              |         |           |
| AMH-CNN                         |           |                 |              |              |         |           |
| SK-CNN                          |           |                 |              |              |         |           |
| Light-MHTCN                     |           |                 |              |              |         |           |
| CNN-HS                          |           |                 |              |              |         |           |
| DeepSense                       |           |                 |              |              |         |           |
| LSTM-RNN                        |           |                 |              |              |         |           |
| Lightweight RNN-LSTM            |           |                 |              |              |         |           |
| PerRNN                          |           |                 |              |              |         |           |
| DCNN                            |           |                 |              |              |         |           |


## Supervised Domain-Adaptation Benchmarks

Trained in one dataset and tested in another dataset, leave one out cross validation style

| **Model**                       | **KuHar** | **MotionSense** | **RW-Waist** | **RW-Thigh** | **UCI** | **WISDM** |
|---------------------------------|-----------|-----------------|--------------|:------------:|:-------:|-----------|
| DeepConvLSTM                    |           |                 |              |              |         |           |
| Simple 1D Convolutional Network |           |                 |              |              |         |           |
| Simple 2D Convolutional Network |           |                 |              |              |         |           |
| CNN_HaEtAl_1D                   |           |                 |              |              |         |           |
| CNN_HaEtAl_2D                   |           |                 |              |              |         |           |
| CNN PF                          |           |                 |              |              |         |           |
| CNN PPF                         |           |                 |              |              |         |           |
| IMU Transformer                 |           |                 |              |              |         |           |
| IMU CNN                         |           |                 |              |              |         |           |
| InceptionTime                   |           |                 |              |              |         |           |
| 1D-ResNet                       |           |                 |              |              |         |           |
| 1D-ResNet-SE-8                  |           |                 |              |              |         |           |
| 1D-ResNet-SE-5                  |           |                 |              |              |         |           |
| InnoHAR                         |           |                 |              |              |         |           |
| TCN                             |           |                 |              |              |         |           |
| MCNN                            |           |                 |              |              |         |           |
| HVMAN                           |           |                 |              |              |         |           |
| EnsemConvNet                    |           |                 |              |              |         |           |
| LW-CNN                          |           |                 |              |              |         |           |
| DanHAR                          |           |                 |              |              |         |           |
| AMH-CNN                         |           |                 |              |              |         |           |
| SK-CNN                          |           |                 |              |              |         |           |
| Light-MHTCN                     |           |                 |              |              |         |           |
| CNN-HS                          |           |                 |              |              |         |           |
| DeepSense                       |           |                 |              |              |         |           |
| LSTM-RNN                        |           |                 |              |              |         |           |
| Lightweight RNN-LSTM            |           |                 |              |              |         |           |
| PerRNN                          |           |                 |              |              |         |           |
| DCNN                            |           |                 |              |              |         |           |


## Self-Supervised Benchmarks

Trained in one dataset and tested in another dataset, leave one out cross validation style

### Pre-training ExtraSensory

Pre-train using ExtraSensory. Fine-tune and test on same dataset



| **Model**                       | **KuHar** | **MotionSense** | **RW-Waist** | **RW-Thigh** | **UCI** | **WISDM** |
|---------------------------------|-----------|-----------------|--------------|:------------:|:-------:|-----------|
| DeepConvLSTM                    |           |                 |              |              |         |           |
| Simple 1D Convolutional Network |           |                 |              |              |         |           |
| Simple 2D Convolutional Network |           |                 |              |              |         |           |
| CNN_HaEtAl_1D                   |           |                 |              |              |         |           |
| CNN_HaEtAl_2D                   |           |                 |              |              |         |           |
| CNN PF                          |           |                 |              |              |         |           |
| CNN PPF                         |           |                 |              |              |         |           |
| IMU Transformer                 |           |                 |              |              |         |           |
| IMU CNN                         |           |                 |              |              |         |           |
| InceptionTime                   |           |                 |              |              |         |           |
| 1D-ResNet                       |           |                 |              |              |         |           |
| 1D-ResNet-SE-8                  |           |                 |              |              |         |           |
| 1D-ResNet-SE-5                  |           |                 |              |              |         |           |
| InnoHAR                         |           |                 |              |              |         |           |
| TCN                             |           |                 |              |              |         |           |
| MCNN                            |           |                 |              |              |         |           |
| HVMAN                           |           |                 |              |              |         |           |
| EnsemConvNet                    |           |                 |              |              |         |           |
| LW-CNN                          |           |                 |              |              |         |           |
| DanHAR                          |           |                 |              |              |         |           |
| AMH-CNN                         |           |                 |              |              |         |           |
| SK-CNN                          |           |                 |              |              |         |           |
| Light-MHTCN                     |           |                 |              |              |         |           |
| CNN-HS                          |           |                 |              |              |         |           |
| DeepSense                       |           |                 |              |              |         |           |
| LSTM-RNN                        |           |                 |              |              |         |           |
| Lightweight RNN-LSTM            |           |                 |              |              |         |           |
| PerRNN                          |           |                 |              |              |         |           |
| DCNN                            |           |                 |              |              |         |           |


### Pre-train Leave-One-Out

Pre-train using leave-one-out. Fine-tune and test on same dataset (out dataset)


| **Model**                       | **KuHar** | **MotionSense** | **RW-Waist** | **RW-Thigh** | **UCI** | **WISDM** |
|---------------------------------|-----------|-----------------|--------------|:------------:|:-------:|-----------|
| DeepConvLSTM                    |           |                 |              |              |         |           |
| Simple 1D Convolutional Network |           |                 |              |              |         |           |
| Simple 2D Convolutional Network |           |                 |              |              |         |           |
| CNN_HaEtAl_1D                   |           |                 |              |              |         |           |
| CNN_HaEtAl_2D                   |           |                 |              |              |         |           |
| CNN PF                          |           |                 |              |              |         |           |
| CNN PPF                         |           |                 |              |              |         |           |
| IMU Transformer                 |           |                 |              |              |         |           |
| IMU CNN                         |           |                 |              |              |         |           |
| InceptionTime                   |           |                 |              |              |         |           |
| 1D-ResNet                       |           |                 |              |              |         |           |
| 1D-ResNet-SE-8                  |           |                 |              |              |         |           |
| 1D-ResNet-SE-5                  |           |                 |              |              |         |           |
| InnoHAR                         |           |                 |              |              |         |           |
| TCN                             |           |                 |              |              |         |           |
| MCNN                            |           |                 |              |              |         |           |
| HVMAN                           |           |                 |              |              |         |           |
| EnsemConvNet                    |           |                 |              |              |         |           |
| LW-CNN                          |           |                 |              |              |         |           |
| DanHAR                          |           |                 |              |              |         |           |
| AMH-CNN                         |           |                 |              |              |         |           |
| SK-CNN                          |           |                 |              |              |         |           |
| Light-MHTCN                     |           |                 |              |              |         |           |
| CNN-HS                          |           |                 |              |              |         |           |
| DeepSense                       |           |                 |              |              |         |           |
| LSTM-RNN                        |           |                 |              |              |         |           |
| Lightweight RNN-LSTM            |           |                 |              |              |         |           |
| PerRNN                          |           |                 |              |              |         |           |
| DCNN                            |           |                 |              |              |         |           |


### Pre-train ExtraSensory Leave-one-out

Pre-train using ExtraSensory. Fine-tune using leave-one-out. Test on same dataset (out dataset)


| **Model**                       | **KuHar** | **MotionSense** | **RW-Waist** | **RW-Thigh** | **UCI** | **WISDM** |
|---------------------------------|-----------|-----------------|--------------|:------------:|:-------:|-----------|
| DeepConvLSTM                    |           |                 |              |              |         |           |
| Simple 1D Convolutional Network |           |                 |              |              |         |           |
| Simple 2D Convolutional Network |           |                 |              |              |         |           |
| CNN_HaEtAl_1D                   |           |                 |              |              |         |           |
| CNN_HaEtAl_2D                   |           |                 |              |              |         |           |
| CNN PF                          |           |                 |              |              |         |           |
| CNN PPF                         |           |                 |              |              |         |           |
| IMU Transformer                 |           |                 |              |              |         |           |
| IMU CNN                         |           |                 |              |              |         |           |
| InceptionTime                   |           |                 |              |              |         |           |
| 1D-ResNet                       |           |                 |              |              |         |           |
| 1D-ResNet-SE-8                  |           |                 |              |              |         |           |
| 1D-ResNet-SE-5                  |           |                 |              |              |         |           |
| InnoHAR                         |           |                 |              |              |         |           |
| TCN                             |           |                 |              |              |         |           |
| MCNN                            |           |                 |              |              |         |           |
| HVMAN                           |           |                 |              |              |         |           |
| EnsemConvNet                    |           |                 |              |              |         |           |
| LW-CNN                          |           |                 |              |              |         |           |
| DanHAR                          |           |                 |              |              |         |           |
| AMH-CNN                         |           |                 |              |              |         |           |
| SK-CNN                          |           |                 |              |              |         |           |
| Light-MHTCN                     |           |                 |              |              |         |           |
| CNN-HS                          |           |                 |              |              |         |           |
| DeepSense                       |           |                 |              |              |         |           |
| LSTM-RNN                        |           |                 |              |              |         |           |
| Lightweight RNN-LSTM            |           |                 |              |              |         |           |
| PerRNN                          |           |                 |              |              |         |           |
| DCNN                            |           |                 |              |              |         |           |


## Domain Generalization Methods


* DDG
* MixUp
* FixMatch
