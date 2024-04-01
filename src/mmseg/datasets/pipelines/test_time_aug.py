import warnings

import mmcv

from src.mmseg.datasets.builder import PIPELINES
from src.mmseg.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """
    Aumento de tiempo de prueba con múltiples escalas y volteo.

    Una configuración de ejemplo es la siguiente:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    Después de MultiScaleFlipAug con la configuración anterior, los resultados están envueltos
    en listas de la misma longitud de la siguiente manera:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transformaciones para aplicar en cada aumento.
        img_scale (None | tuple | list[tuple]): Escalas de imagen para el redimensionamiento.
        img_ratios (float | list[float]): Razones de imagen para el redimensionamiento
        flip (bool): Si aplicar el aumento de volteo. Por defecto: False.
        flip_direction (str | list[str]): Direcciones de aumento de volteo,
            las opciones son "horizontal" y "vertical". Si flip_direction es una lista,
            se aplicarán múltiples aumentos de volteo.
            No tiene efecto cuando flip == False. Por defecto: "horizontal".
    """
    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # Modo 1: Dado img_scale=None y un rango de ratio de imagen
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # Modo 2: Dado una escala y un rango de ratio de imagen
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # Modo 3: Dadas múltiples escalas
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')


    def __call__(self, results):
        """
        Llama a la función para aplicar transformaciones de aumento de tiempo de prueba a los resultados.

        Args:
            results (dict): Diccionario de resultados que contiene los datos para transformar.

        Returns:
           dict[str: list]: Los datos aumentados, donde cada valor está envuelto
               en una lista.
        """
        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # Lista de diccionarios a diccionario de listas
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
