from abc import ABC, abstractmethod

class Configurable:
    @abstractmethod
    def get_config(self) -> dict:
        raise NotImplementedError