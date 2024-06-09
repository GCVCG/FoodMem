import os.path as osp

import mmcv
import numpy as np

from src.mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """
    Loads an image from a file.

    The required keys are "img_prefix" and "img_info" (a dictionary that must contain the
    key "filename"). The added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (equal to `img_shape`), "pad_shape" (equal to `img_shape`),
    "scale_factor" (1.0), and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32 numpy array.
            If set to False, the loaded image is a uint8 array. Default: False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Default: 'color'.
        file_client_args (dict): Arguments for instantiating a FileClient.
            See :class:`mmcv.fileio.FileClient` for more details.
            Default: ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend


    def __call__(self, results):
        """
        Calls functions to load the image and get image meta-information.

        Args:
            results (dict): Results dictionary from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: Dictionary containing the loaded image and image meta-information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"

        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """
    Loads annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether to reduce all label values by 1.
            Usually used for datasets where 0 is the background label.
            Default: False.
        file_client_args (dict): Arguments for instantiating a FileClient.
            See :class:`mmcv.fileio.FileClient` for more details.
            Default: ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """
        Calls the function to load annotations of various types.

        Args:
            results (dict): Results dictionary from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: Dictionary containing the loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if len(gt_semantic_seg.shape) == 3:
            gt_semantic_seg = gt_semantic_seg[:,:,2]

        # Modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id

        # Reduce zero_label
        if self.reduce_zero_label:
            # Avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"

        return repr_str

