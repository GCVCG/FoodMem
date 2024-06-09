import warnings

import mmcv

from src.mmseg.datasets.builder import PIPELINES
from src.mmseg.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """
    Test-time augmentation with multiple scales and flipping.

    An example configuration is as follows:

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

    After MultiScaleFlipAug with the above configuration, the results are wrapped
    in lists of the same length as follows:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transformations to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Image scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing.
        flip (bool): Whether to apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is a list,
            multiple flip augmentations will be applied. It has no effect when flip == False. Default: "horizontal".
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
            # Mode 1: Given img_scale=None and an image ratio range
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # Mode 2: Given a scale and an image ratio range
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # Mode 3: Given multiple scales
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
        Calls the function to apply test-time augmentation transformations to the results.

        Args:
            results (dict): Results dictionary containing the data to transform.

        Returns:
           dict[str: list]: Augmented data, where each value is wrapped
               in a list.
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

        # List of dictionaries to dictionary of lists
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
