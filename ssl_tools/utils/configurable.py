from abc import ABC, abstractmethod


class Configurable:
    """Configurable interface for models and other objects that can be 
    configured with a dictionary. For now, this interface is used to save the hyperparameters of the models.
    """

    @abstractmethod
    def get_config(self) -> dict:
        raise NotImplementedError
