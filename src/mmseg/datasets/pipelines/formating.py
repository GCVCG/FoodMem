from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from src.mmseg.datasets.builder import PIPELINES


def to_tensor(data):
    """
    Convierte objetos de varios tipos de Python a :obj:`torch.Tensor`.

    Se admiten los siguientes tipos: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` y :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Datos a
            ser convertidos.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor(object):
    """
    Convierte algunos resultados a :obj:`torch.Tensor` mediante claves dadas.

    Args:
        keys (Sequence[str]): Claves que deben convertirse a Tensor.
    """
    def __init__(self, keys):
        self.keys = keys


    def __call__(self, results):
        """
        Función de llamada para convertir datos en resultados a :obj:`torch.Tensor`.

        Args:
            results (dict): Diccionario de resultados que contiene los datos a convertir.

        Returns:
            dict: El diccionario de resultados contiene los datos convertidos
                a :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor(object):
    """
    Convierte la imagen a :obj:`torch.Tensor` mediante claves dadas.

    El orden de dimensiones de la imagen de entrada es (H, W, C). El pipeline convertirá
    a (C, H, W). Si solo se proporcionan 2 dimensiones (H, W), la salida sería
    (1, H, W).

    Args:
        keys (Sequence[str]): Clave de las imágenes que se convertirán a Tensor.
    """
    def __init__(self, keys):
        self.keys = keys


    def __call__(self, results):
        """
        Llama a la función para convertir la imagen en los resultados a :obj:`torch.Tensor` y
        transpone el orden de los canales.

        Args:
            results (dict): Diccionario de resultados que contiene los datos de imagen a convertir.

        Returns:
            dict: El diccionario de resultados contiene la imagen convertida
                a :obj:`torch.Tensor` y transpuesta a un orden (C, H, W).
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Transpose(object):
    """
    Transpone algunos resultados mediante claves dadas.

    Args:
        keys (Sequence[str]): Claves de los resultados que se transpondrán.
        order (Sequence[int]): Orden de transposición.
    """
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order


    def __call__(self, results):
        """
        Llama a la función para convertir la imagen en los resultados a :obj:`torch.Tensor` y
        transpone el orden de los canales.

        Args:
            results (dict): Diccionario de resultados que contiene los datos de imagen a convertir.

        Returns:
            dict: El diccionario de resultados contiene la imagen convertida
                a :obj:`torch.Tensor` y transpuesta a un orden (C, H, W).
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results


    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer(object):
    """
    Convierte los resultados a :obj:`mmcv.DataContainer` mediante campos dados.

    Args:
        fields (Sequence[dict]): Cada campo es un diccionario como
            ``dict(key='xxx', **kwargs)``. La ``key`` en el resultado se
            convertirá a :obj:`mmcv.DataContainer` con ``**kwargs``.
            Por defecto: ``(dict(key='img',
            stack=True), dict(key='gt_semantic_seg'))``.
    """
    def __init__(self,
                 fields=(dict(key='img',
                              stack=True), dict(key='gt_semantic_seg'))):
        self.fields = fields


    def __call__(self, results):
        """
        Llama a la función para convertir los datos en los resultados a
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Diccionario de resultados que contiene los datos a convertir.

        Returns:
            dict: El diccionario de resultados contiene los datos convertidos a
                :obj:`mmcv.DataContainer`.
        """
        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """
    Paquete de formato predeterminado.

    Simplifica el pipeline de formato de campos comunes, incluidos "img"
    y "gt_semantic_seg". Estos campos se formatean de la siguiente manera.

    - img: (1)transpuesto, (2) a tensor, (3) a DataContainer (stack=True)
    - gt_semantic_seg: (1)dimensión sin comprimir dim-0 (2)a tensor,
                       (3)a DataContainer (stack=True)
    """
    def __call__(self, results):
        """
        Llama a la función para transformar y formatear campos comunes en los resultados.

        Args:
            results (dict): Diccionario de resultados que contiene los datos a convertir.

        Returns:
            dict: El diccionario de resultados contiene los datos que se formatean con
                el paquete predeterminado.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results


    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect(object):
    """
    Recopila datos del cargador relevantes para la tarea específica.

    Esta suele ser la última etapa del pipeline del cargador de datos. 
    Típicamente, las claves se establecen en un subconjunto de "img", "gt_semantic_seg".

    El elemento "img_meta" siempre se completa. El contenido del diccionario "img_meta"
    depende de "meta_keys". Por defecto, esto incluye:

        - "img_shape": forma de la imagen de entrada a la red como una tupla
            (h, w, c). Tenga en cuenta que las imágenes pueden tener relleno de ceros en la parte inferior/derecha
            si el tensor del lote es más grande que esta forma.

        - "scale_factor": un flotante que indica la escala de preprocesamiento

        - "flip": un booleano que indica si se utilizó una transformación de volteo de imagen

        - "filename": ruta al archivo de imagen

        - "ori_shape": forma original de la imagen como una tupla (h, w, c)

        - "pad_shape": forma de la imagen después del relleno

        - "img_norm_cfg": un diccionario de información de normalización:
            - media - sustracción de media por canal
            - std - divisor de std por canal
            - to_rgb - booleano que indica si bgr se convirtió a rgb

    Args:
        keys (Sequence[str]): Claves de los resultados que se recopilarán en ``data``.
        meta_keys (Sequence[str], opcional): Claves meta que se convertirán en
            ``mmcv.DataContainer`` y se recopilarán en ``data[img_metas]``.
            Por defecto: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """
        Llama a la función para recopilar claves en los resultados. Las claves en ``meta_keys``
        se convertirán en :obj:mmcv.DataContainer.

        Args:
            results (dict): Diccionario de resultados que contiene los datos a recopilar.

        Returns:
            dict: El diccionario de resultados contiene las siguientes claves
                - claves en ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
