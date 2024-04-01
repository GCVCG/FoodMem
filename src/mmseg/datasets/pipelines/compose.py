import collections

from mmcv.utils import build_from_cfg

from src.mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """
    Compone múltiples transformaciones secuencialmente.

    Args:
        transforms (Sequence[dict | callable]): Secuencia de objetos de transformación o
            configuraciones de diccionario para componer.
    """
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')


    def __call__(self, data):
        """
        Llama a la función para aplicar las transformaciones secuencialmente.

        Args:
            data (dict): Un diccionario de resultados que contiene los datos para transformar.

        Returns:
           dict: Datos transformados.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
