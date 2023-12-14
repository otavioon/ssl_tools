# Experiments

This directory contains the experiments for the machine learning tasks.
Each subdirectory contains the experiments for a specific task.

## Creating PyTorch Lightning Experiments

The `ligthning_cli.py` file contains base classes to creating experiments using 
PyTorch Lightning. In summary, the classes are data classes, which hold 
necessary information for training and testing a model. The classes are:

- `LightningTrain`: Defines the common parameters used to train a model, such as 
batch size, learning rate, number of workers, etc.
- `LightningTest`: Defines the common parameters used to test a model, such as
batch size, number of workers, etc.


These classes were develop either to store the parameters for the experiments
and also to be used using the `jsonargparse` CLI, which allows to create
command line interfaces using data classes. Every parameter in the class 
constructor is a parameter that can be passed to the CLI. 
Thus, if we inherit from these classes, our CLI interface will have all the
parameters defined in the base class, plus the parameters defined in the
child class.

For example, if we create a file names `test.py`, which inherits from 
`LightningTrain` class. The file should look like this:

```python
from lightning_cli import LightningTrain
from jsonargparse import CLI

class Test(LightningTrain):
    def __init__(self, my_param: int = 1, *args, **kwargs):
        """All parameters defined in the base class are available here. You can
        add new parameters here. The parameters defined here will be available
        in the command-line interface.
        """
        super().__init__(*args, **kwargs)
        self.my_param = my_param

    def __call__(self):
        """This method is called when the CLI is executed. Your code goes here. 
        """
        # This is parameters from the child class
        print(f"My param: {self.my_param}")
        # We can access parameters from the base class
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Log dir: {self.log_dir}")
        print(f"Experiment name: {self.experiment_name}")
        print(f"Experiment version: {self.experiment_version}")
        

if __name__ == '__main__':
    # Create a command line interface for Test class
    CLI(Test, as_positional=False)()
```

All parameters for `__init__` methods (base classes and child classes) turns 
into parameters for the CLI. The `__call__` method is called when the CLI is
executed. 

If we run the command `python test.py --help`, we will see the following output:

```bash
usage: test.py [-h] [--config CONFIG] [--print_config[=flags]] [--my_param MY_PARAM] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--log_dir LOG_DIR] [--name NAME]
               [--version VERSION] [--load LOAD] [--checkpoint_metric CHECKPOINT_METRIC]
               [--checkpoint_metric_mode CHECKPOINT_METRIC_MODE] [--accelerator ACCELERATOR] [--devices DEVICES]
               [--strategy STRATEGY] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES]
               [--num_nodes NUM_NODES] [--num_workers NUM_WORKERS] [--seed SEED]

All parameters defined in the base class are available here. You can

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes
                        the output and are one or more keywords separated by comma. The supported flags are: comments,
                        skip_default, skip_null.
  --my_param MY_PARAM   (type: int, default: 1)
  --epochs EPOCHS       Number of epochs to pre-train the model (type: int, default: 1)
  --batch_size BATCH_SIZE
                        The batch size (type: int, default: 1)
  --learning_rate LEARNING_RATE
                        The learning rate of the optimizer (type: float, default: 0.001)
  --log_dir LOG_DIR     Path to the location where logs will be stored (type: str, default: logs)
  --name NAME           The name of the experiment (it will be used to compose the path of the experiments, such as logs and
                        checkpoints) (type: Optional[str], default: null)
  --version VERSION     The version of the experiment. If not is provided the current date and time will be used as the
                        version (type: Union[str, int, null], default: null)
  --load LOAD           The path to a checkpoint to load (type: Optional[str], default: null)
  --checkpoint_metric CHECKPOINT_METRIC
                        The metric to monitor for checkpointing. If not provided, the last model will be saved (type:
                        Optional[str], default: null)
  --checkpoint_metric_mode CHECKPOINT_METRIC_MODE
                        The mode of the metric to monitor (min, max or mean). Defaults to "min" (type: str, default: min)
  --accelerator ACCELERATOR
                        The accelerator to use. Defaults to "cpu" (type: str, default: cpu)
  --devices DEVICES     The number of devices to use. Defaults to 1 (type: int, default: 1)
  --strategy STRATEGY   The strategy to use. Defaults to "auto" (type: str, default: auto)
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        The number of batches to use for training. Defaults to 1.0 (use all batches) (type: Union[float,
                        int], default: 1.0)
  --limit_val_batches LIMIT_VAL_BATCHES
                        The number of batches to use for validation. Defaults to 1.0 (use all batches) (type: Union[float,
                        int], default: 1.0)
  --num_nodes NUM_NODES
                        The number of nodes to use. Defaults to 1 (type: int, default: 1)
  --num_workers NUM_WORKERS
                        The number of workers to use for the dataloader. (type: Optional[int], default: null)
  --seed SEED           The seed to use. (type: Optional[int], default: null)

```

We can see that all parameters defined in the base class are available in the
CLI. We can also see that the parameters defined in the child class are also
available.

If we run the command `python test.py --my_param 2 --batch_size 32`, we will
see the following output:

```bash
My param: 2
Batch size: 32
Learning rate: 0.001
Log dir: logs
Experiment name: None
Experiment version: 2023-12-14_12-28-37
```

Thus, experiments can be launched from the CLI or we can instantiate inside any 
other python script.

