import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from src.mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Resize(object):
    """
    Resize images and segmentations.

    This transformation resizes the input image to a specified scale. If the input dictionary
    contains the "scale" key, then the scale in the input dictionary is used,
    otherwise the scale specified in the init method is used.

    ``img_scale`` can be None, a tuple (single scale), or a list of tuples
    (multiple scales). There are 4 modes of multiple scales:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of the image in the results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single scale), a ratio from
        the ratio range is randomly sampled and multiplied with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: a scale from a range is randomly sampled. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: a scale from multiple scales is randomly sampled. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Image scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to maintain the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # Mode 1: Given img_scale=None and an image ratio range
            # Mode 2: Given a scale and an image ratio range
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # Mode 3 and 4: Given multiple scales or a scale range
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """
        Randomly selects an image scale from the given candidates.

        Args:
            img_scales (list[tuple]): Image scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_idx)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """
        assert mmcv.is_list_of(img_scales, tuple)

        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]

        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """
        Randomly samples an image scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Image scale range to sample from.
                There should be two tuples in img_scales, specifying the lower bound
                and upper bound of the image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is the sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2

        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]

        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)

        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)

        img_scale = (long_edge, short_edge)

        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """
        Randomly samples an image scale when ``ratio_range`` is specified.

        It randomly samples a ratio from the range specified by
        ``ratio_range``. It would then be multiplied with ``img_scale`` to
        generate the sampled scale.

        Args:
            img_scale (tuple): Base image scale to multiply with the ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is the sampled ratio multiplied by ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2

        min_ratio, max_ratio = ratio_range

        assert min_ratio <= max_ratio

        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)

        return scale, None

    def _random_scale(self, results):
        """
        Randomly samples an image scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio would be sampled and
        multiplied with ``img_scale``.
        If multiple scales are specified via ``img_scale``, a scale would be
        sampled according to ``multiscale_mode``.
        Otherwise, a single scale would be used.

        Args:
            results (dict): Dictionary of results from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added in
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx


    def _resize_img(self, results):
        """Resizes images with ``results['scale']``."""

        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # The w_scale and h_scale have minor difference
            # real correction should be made in the future in mmcv.imrescale
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # In case there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio


    def _resize_seg(self, results):
        """Resizes semantic segmentation map with ``results['scale']``."""

        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg


    def __call__(self, results):
        """
        Call function to resize images, bounding boxes, masks, semantic segmentation.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Resized results, keys 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' are added to the result dictionary.
        """

        if 'scale' not in results:
            self._random_scale(results)

        self._resize_img(results)
        self._resize_seg(results)

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')

        return repr_str



@PIPELINES.register_module()
class RandomFlip(object):
    """
    Flips the image and segmentation.

    If the input dictionary contains the "flip" key, then the flag will be used,
    otherwise, it will be randomly decided according to a specified probability in the init method.

    Args:
        prob (float, optional): The probability of flipping. Default: None.
        direction(str, optional): The direction of flipping. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction

        if prob is not None:
            assert prob >= 0 and prob <= 1

        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Calls function to flip bounding boxes, masks, semantic segmentation maps.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Flipped results, keys 'flip', 'flip_direction' are added to the
                result dictionary.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip

        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction

        if results['flip']:
            # Flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # Flip segmentations
            for key in results.get('seg_fields', []):
                # Use copy() to make numpy step positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Pad(object):
    """
    Pads the image and mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Keys "pad_shape", "pad_fixed_size", "pad_size_divisor" are added.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of the padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of the segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        # Only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pads the images according to ``self.size``."""

        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pads the masks according to ``results['pad_shape']``."""

        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """
        Calls function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Updated result dictionary.
        """

        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'

        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """
    Normalizes the image.

    Key "img_norm_cfg" is added.

    Args:
        mean (sequence): Mean values of the 3 channels.
        std (sequence): Standard deviation values of the 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is True.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """
        Calls function to normalize images.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Normalized results, key 'img_norm_cfg' is added to the
                results dictionary.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'

        return repr_str


@PIPELINES.register_module()
class Rerange(object):
    """
    Rescales the pixel values of the image.

    Args:
        min_value (float or int): Minimum value of the rescaled image.
            Default: 0.
        max_value (float or int): Maximum value of the rescaled image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """
        Calls function to rescale images.

        Args:
            results (dict): Dictionary of results from loading pipeline.
        Returns:
            dict: Rescaled results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value

        # Rescale to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)

        # Rescale to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'

        return repr_str


@PIPELINES.register_module()
class CLAHE(object):
    """
    Utilizes the CLAHE method to process the image.

    Refer to `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            The input image will be divided into equally sized rectangles.
            Defines the number of rectangles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))

        self.clip_limit = clip_limit

        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2

        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        """
        Calls function to utilize the CLAHE method to process images.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, ' \
                    f'tile_grid_size={self.tile_grid_size})'

        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """
    Randomly crops the image and segmentation.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio a single category could occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0

        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly obtains a crop box."""

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crops from ``img``"""

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        return img

    def __call__(self, results):
        """
        Calls function to randomly crop images and semantic segmentation maps.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in the results dictionary is updated according to the crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)

        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # Crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # Crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate(object):
    """
    Rotate the image and segmentation.

    Args:
        prob (float): Probability of rotation.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of a tuple like (min, max),
            the range of degrees will be (-degree, +degree)
        pad_val (float, optional): Padding value of the image. Default: 0.
        seg_pad_val (float, optional): Padding value of the segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of rotation on
            the source image. If not specified, the center of the image will be used.
            Default: None.
        auto_bound (bool): Whether to adjust the size of the image to cover the entire
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob

        assert prob >= 0 and prob <= 1

        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree

        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """
        Calls function to rotate the image, segmentation maps.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))

        if rotate:
            # Rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pad_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # Rotate segmentation
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pad_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'

        return repr_str


@PIPELINES.register_module()
class RGB2Gray(object):
    """
    Convert an RGB image to a grayscale image.

    This transformation computes the weighted mean of the input image channels with
    ``weights`` and then expands the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    the input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transformation. Default: None.
        weights (tuple[float]): The weights for computing the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):

        assert out_channels is None or out_channels > 0

        self.out_channels = out_channels

        assert isinstance(weights, tuple)

        for item in weights:
            assert isinstance(item, (float, int))

        self.weights = weights


    def __call__(self, results):
        """
        Calls function to convert an RGB image to a grayscale image.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Dictionary of results with grayscale image.
        """

        img = results['img']

        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)

        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)

        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'

        return repr_str


@PIPELINES.register_module()
class AdjustGamma(object):
    """
    Utilizes gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0

        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def __call__(self, results):
        """Calls the function to process the image with gamma correction.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(gamma={self.gamma})'


@PIPELINES.register_module()
class SegRescale(object):
    """
    Rescales the semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor for the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """
        Calls the function to scale the semantic segmentation map.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Dictionary of results with the scaled semantic segmentation map.
        """

        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """
    Applies photometric distortion to the image sequentially, each transformation
    is applied with a probability of 0.5. The position of the random contrast is either in
    the second or the penultimate place.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiply by alpha and add beta with clipping."""

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""

        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))

        return img

    def contrast(self, img):
        """Contrast distortion."""

        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))

        return img

    def saturation(self, img):
        """Saturation distortion."""

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)

        return img

    def hue(self, img):
        """Hue distortion."""

        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)

        return img

    def __call__(self, results):
        """
        Calls the function to perform photometric distortion on the images.

        Args:
            results (dict): Dictionary of results from loading pipeline.

        Returns:
            dict: Dictionary of results with distorted images.
        """
        img = results['img']

        # Random brightness
        img = self.brightness(img)

        # Mode == 0 --> Do random contrast first
        # Mode == 1 --> Do random contrast at the end
        mode = random.randint(2)

        if mode == 1:
            img = self.contrast(img)

        # Random saturation
        img = self.saturation(img)

        # Random hue
        img = self.hue(img)

        # Random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')

        return repr_str

