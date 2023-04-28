# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from mmrotate.models.detectors import dft
import numpy
import torch.fft as fft
import torch.nn as nn


class SpatialCausality(nn.Module):
    def __init__(self, in_channels):
        super(SpatialCausality, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        shape = x.shape
        fmx = fft.fftn(x, dim=(1, 2), norm='ortho')
        fmx_conj = torch.conj(fmx)
        attention = fmx * fmx_conj
        x = x + 0.01 * attention.abs() * x
        return x.float()


class DisEntangle(nn.Module):
    def __init__(self, in_channels):
        super(DisEntangle, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        shape = x.shape
        device = x.device
        frequencies = numpy.fft.fftfreq(shape[1])
        frequencies = torch.tensor(frequencies).to(device)
        fft_compute = fft.fftn(x, dim=1, norm='ortho').abs()
        frequencies = frequencies.unsqueeze(1)
        frequencies = frequencies.unsqueeze(1)
        frequencies = frequencies.unsqueeze(0)
        # frequencies = frequencies.unsqueeze(0)

        x = x + frequencies * frequencies * fft_compute * fft_compute
        return x.float()


@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetectorProgress(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 pro_roi_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetectorProgress, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        if pro_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            pro_roi_head.update(train_cfg=rcnn_train_cfg)
            pro_roi_head.update(test_cfg=test_cfg.rcnn)
            pro_roi_head.pretrained = pretrained
            self.pro_roi_head = build_head(pro_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.disentangle = DisEntangle(in_channels=512)
        self.spatial = SpatialCausality(in_channels=512)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_pro_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'pro_roi_head') and self.pro_roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_dom_labels,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        # img_l, img_h = dft.DFTImg(img.clone())
        x = self.extract_feat(img)
        # y = list(x)
        # for i in range(len(y)):
        #     y[i] = self.spatial(y[i]) + self.disentangle(y[i])
        # # x = self.spatial(x) + self.disentangle(x)
        # x = tuple(y)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)

        else:
            proposal_list = proposals

        roi_losses, bbox_results = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                               gt_bboxes, gt_dom_labels,
                                                               gt_bboxes_ignore, gt_masks,
                                                               **kwargs)

        losses.update(roi_losses)

        pro_roi_losses = self.pro_roi_head.forward_train(x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels, bbox_results,
                                                         gt_bboxes_ignore, gt_masks,
                                                         **kwargs)
        losses.update(pro_roi_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        bbox_results, coarse_results = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        bbox_results= self.pro_roi_head.simple_test(
            x, proposal_list, img_metas, bbox_results, coarse_results, rescale=rescale)
        return bbox_results

        # return self.roi_head.simple_test(
        #     x_l, proposal_list_l, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
