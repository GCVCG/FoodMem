from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Clase base del muestreador de píxeles."""

    def __init__(self, **kwargs):
        pass


    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Marcador de posición para la función de muestreo."""
        pass
