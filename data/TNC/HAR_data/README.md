# UCI-HAR Preprocessed data

Este diretório contém os dados pré-processados do conjunto de dados UCI-HAR. Os dados foram pré-processados utilizando o `script` [`data/HAR_data.py`](https://github.com/sanatonek/TNC_representation_learning/blob/master/data/HAR_data.py) disponibilizado no repositório do projeto [TNC_representation_learning](https://github.com/sanatonek/TNC_representation_learning). 
Estes dados são utilizados para reprodução do trabalho intitulado [Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding](https://openreview.net/forum?id=8qDwejCuCN). A implementação dos modelos TNC e CPC foram obtidas deste trabalho e estes são os dados foram utilizados para reproduzir os resultados obtidos pelos autores.

Segundo o trabalho dos autores, na seção 3.3 (HUMAN ACTIVITY RECOGNITION (HAR) DATA), sobre os dados de HAR:

> "Human Activity Recognition (HAR) is the problem of predicting the type of activity using temporal data from accelerometer and gyroscope measurements. We use the HAR dataset from the UCI Machine Learning Repository 2 that includes data collected from 30 individuals using a smartwatch. Each person performs six activities: 1) walking, 2) walking upstairs, 3) walking downstairs, 4) sitting, 5) standing, and 6) laying. The time-series measurements are pre-processed to extract 561 features. For our purpose, we concatenate the activity samples from every individual over time using the subject identifier to build the full-time series for each subject, which includes continuous activity change. Similar to the simulated data setting, we use a single-layer RNN encoder. The selected window size is 4, representing about 15 seconds of recording, and the representations are encoded in a 10-dimensional vector space"

Os dados foram processados e armazenados no formato numpy. Este diretório contém os seguintes arquivos:
* `x_train.npy`: Dados de treino, com dimensão (21, 561, 281);
* `y_train.npy`: Rótulos dos dados de treino, com dimensão (21, 281);
* `x_test.npy`: Dados de teste, com dimensão (9, 561, 288);
* `y_test.npy`: Rótulos dos dados de teste, com dimensão (9, 288).

Nos arquivos de treino, a primeira dimensão são as amostras (atividades concatenadas por indivíduos, para cada um dos 21 indivíduos), a segunda dimensão são as características (561) e a terceira dimensão são as janelas de tempo (281). Nos arquivos de teste, a primeira dimensão são as amostras (atividades concatenadas por indivíduos dos 9 indivíduos), a segunda dimensão são as janelas de tempo (288).

Para carregar um arquivo, basta utilizar a função `numpy.load`:

```python
import numpy as np
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
```

As variáveis são do tipo `numpy.ndarray`.