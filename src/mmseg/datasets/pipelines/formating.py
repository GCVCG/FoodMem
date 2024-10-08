from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from src.mmseg.datasets.builder import PIPELINES


def to_tensor(data):
    """
    Converts objects of various Python types to :obj:`torch.Tensor`.

    The following types are supported: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`, and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
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
    Converts some results to :obj:`torch.Tensor` given keys.

    Args:
        keys (Sequence[str]): Keys that should be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """
        Callable function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Dictionary of results containing data to convert.

        Returns:
            dict: Dictionary of results containing data converted to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            results[key] = to_tensor(results[key])

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor(object):
    """
    Converts the image to :obj:`torch.Tensor` given keys.

    The input image dimensions order is (H, W, C). The pipeline will convert it to (C, H, W).
    If only 2 dimensions (H, W) are provided, the output would be (1, H, W).

    Args:
        keys (Sequence[str]): Key of the images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """
        Callable function to convert the image in the results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Dictionary of results containing image data to convert.

        Returns:
            dict: Dictionary of results containing the image converted to :obj:`torch.Tensor`
                and transposed to an order (C, H, W).
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
    Transposes some results given keys.

    Args:
        keys (Sequence[str]): Keys of the results to be transposed.
        order (Sequence[int]): Transpose order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """
        Callable function to transpose the data in the results dictionary.

        Args:
            results (dict): Dictionary of results containing data to be transposed.

        Returns:
            dict: Dictionary of results containing the data transposed according to the specified order.
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
    Converts the results to :obj:`mmcv.DataContainer` using given fields.

    Args:
        fields (Sequence[dict]): Each field is a dictionary like
            ``dict(key='xxx', **kwargs)``. The ``key`` in the result will be
            converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True), dict(key='gt_semantic_seg'))``.
    """

    def __init__(self,
                 fields=(dict(key='img',
                              stack=True), dict(key='gt_semantic_seg'))):
        self.fields = fields

    def __call__(self, results):
        """
        Callable function to convert the data in the results to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Dictionary of results containing the data to be converted.

        Returns:
            dict: Dictionary of results containing the data converted to :obj:`mmcv.DataContainer`.
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
    Default formatting bundle.

    Simplifies the formatting pipeline for common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1) transposed, (2) to tensor, (3) to DataContainer (stack=True)
    - gt_semantic_seg: (1) uncompressed dim-0 dimension, (2) to tensor,
                       (3) to DataContainer (stack=True)
    """

    def __call__(self, results):
        """
        Callable function to transform and format common fields in the results.

        Args:
            results (dict): Dictionary of results containing the data to be formatted.

        Returns:
            dict: Dictionary of results containing the data formatted with
                the default bundle.
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
    Collects relevant loader data for the specific task.

    This is typically the last stage of the data loader pipeline.
    Typically, keys are set to a subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated. The content of the "img_meta" dictionary
    depends on "meta_keys". By default, this includes:

        - "img_shape": shape of input image to the network as a tuple
            (h, w, c). Note that images may have zero padding at the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if an image flip transformation was used

        - "filename": path to image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": shape of the image after padding

        - "img_norm_cfg": a dictionary of normalization information:
            - mean - per-channel mean subtraction
            - std - per-channel std divisor
            - to_rgb - boolean indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys from the results to be collected into ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected into ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
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
        Callable function to collect keys into results. Keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Dictionary of results containing the data to be collected.

        Returns:
            dict: Dictionary of results containing the following keys
                - keys in ``self.keys``
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

