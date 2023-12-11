# Self-Supervised Learning Tools

This repository contains tools for performing experiments with self-supervised 
learning (SSL) methods. This standartized framework allows to easily compare 
different methods and to reproduce results from the papers. It is based on the 
[PyTorch Lightning](https://lightning.ai/) and implements some SSL methods, 
including:

- [Contrastive Predictive Coding (CPC)](https://arxiv.org/abs/1807.03748)   [DEMO]()
- [Temporal Neighbourhood Coding (TNC)](https://arxiv.org/abs/2106.00750)   [DEMO]()
- [Time-Frequency Consistency (TFC)](https://arxiv.org/abs/2206.08496)      [DEMO]()

Furthermore it implements all necessary components for training and evaluating,
including: transforms, data loaders, models, losses, optimizers, among others.
Also, it provides a simple interface for adding new methods and scripts to 
reproduce the experiments.

## Installation

You may install the package using pip. Since the package is not yet available
on PyPI, you may install it directly from GitHub:

```bash
pip install git+https://github.com/otavioon/ssl_tools.git
```

Orm for development, you may clone the repository and install it from the source:

```bash
git clone https://github.com/otavioon/ssl_tools.git
cd ssl_tools
pip install -e .
```

## Main Components

The framework is decoupled in different components to allow easy customization
and extension. The main components are:

- **Dataset**

    The dataset is responsible for loading the data and applying the transforms
    to the samples. For now, we use 
    [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets),
    which implements the `__getitem__` and `__len__` methods. 

- **Data Loader**

    The data loader encapsulates the dataset and is responsible for creating the
    batches. In general, we use the torch's default 
    [data loader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

- **Data Module**

    The data module is responsible for creating the data loaders for the 
    training, validation and test sets. It also defines the transforms that 
    will be applied to the samples, depending on the split (train, val, test).
    This component is used to train any model, not only the SSL methods. It is 
    implemented using Pytorch Lightning Data Module and is used to allow 
    reproducibility of the experiments.

- **Backbone Network**

    The backbone network the neural network that will be trained using the
    self-supervised learning method, that is, it will be trained for the 
    pretext task (the task that does not require annotated data). This network
    will be used to extract features from the input data, which later will be
    used to train the downstream task. The main backbone module is implemented 
    using Pytorch Lightning.

- **Head**

    The heads represent the final layers integrated onto the neural network, 
    placed atop the backbone. These components transform the backbone's 
    outputs, often referred to as embeddings, representations, or features, 
    into a fresh space for loss calculation. We offers standard heads that can 
    be seamlessly appended to any backbone.

- **Transforms**

    The transforms are applied to the samples before they are fed to the 
    backbone network. They are used to augment the data and to create the 
    pretext task. We offer some standard transforms, but you can easily create
    your own. Transforms must implement the `__call__` method.

- **Losses**

    Self-supervised learning methods usually use different losses than the
    supervised methods. We offer some standard losses, but you can easily
    create your own.

- **Experiment**

    The experiment is responsible for training the model and/or evaluating it. 
    It is composed of the data module, the backbone network and the head. It 
    provides an interface to fetch the data loaders, train the model, evaluate
    it and save/load the model's weights, as well as to log the results. This
    allows to easily compare different methods and to reproduce the experiments.

- **Analysis**

    The analysis is responsible for analyzing the results of the experiments.
    It provides an interface to load the results, plot the results and save
    the plots. This allows to easily compare different methods and to reproduce
    the experiments.

## Organization

The repository is organized as follows:

```
ssl_tools
├── analysis
├── callbacks
├── data
│   ├── data_modules
│   └── datasets
├── experiments
│   └── task_1
│      ├── scripts
│      └── logs
├── losses
├── models
│   ├── layers
│   ├── nets
│   └── ssl
├── transforms
└── utils

```

* **analysis**

    Contains the analysis module, which is responsible for analyzing the 
    results of the experiments.

* **callbacks**

    Contains the callbacks that can be used during the training of the models.
    Custom callbacks can be easily created by extending the Lightning 
    `Callback` class.

* **data/data_modules**

    Contains the data modules, which are responsible for creating the data 
    loaders for the training, validation and test sets. It also defines the 
    transforms that will be applied to the samples, depending on the split 
    (train, val, test).

* **data/datasets**

    Contains the datasets, which are responsible for loading the data and 
    applying the transforms to the samples. Note all operations are performed 
    using numpy arrays, and the samples are converted to tensors only when they 
    are fed to the neural network, automatically by the data loader. Thus,
    the datasets must work with numpy arrays.

* **experiments**

    Contains the experiments, which are responsible for training the model 
    and/or evaluating it. In general, we have one experiment for each task, 
    which is inside a folder with the task's name. Each experiment folder
    contains the scripts to train and evaluate the model, as well as the
    logs with the results.

* **losses**

    Contains the losses, which are used to train the models. Losses may be a 
    callable object or a ``torch.nn.Module``.

* **models/layers**

    Contains the layers, which are used to create the neural networks. Layers are
    usually ``torch.nn.Module`` or ``lightning.LightningModule.``

* **models/nets**

    Contains custom neural networks. They may be used as backbones.

* **models/ssl**

    Contains the self-supervised learning methods, which are used to train the
    backbone networks. Also, it contains the heads, which are used to transform
    the backbone's outputs into a fresh space for loss calculation. Note that, 
    default authors' backbone is implemented here. 

* **transforms**

    Implement the transforms, which is used to augment the data and to create
    the pretext task. Transforms must implement the `__call__` method.

* **utils**

    Contains some utility functions.


## Usage

[TODO]

## Contributing

[TODO]

## License

[MIT](https://choosealicense.com/licenses/mit/)