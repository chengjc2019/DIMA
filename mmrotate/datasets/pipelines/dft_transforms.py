import inspect
import warnings

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import ROTATED_PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@ROTATED_PIPELINES.register_module()
class DftNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.

        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        for key in results.get('img_l_fields', ['img_l']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        for key in results.get('img_h_fields', ['img_h']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class DftPad:

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        pad_val_l = self.pad_val.get('img_l', 0)
        pad_val_h = self.pad_val.get('img_h', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        for key in results.get('img_fields', ['img_l']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val_l)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val_l)
            results[key] = padded_img
        for key in results.get('img_fields', ['img_h']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val_h)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val_h)
            results[key] = padded_img

        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):

        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
