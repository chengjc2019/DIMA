# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule
import torch
import torch.fft as fft
import numpy as np


# from dft import DFTImg


@ROTATED_DETECTORS.register_module()
class DFTS2ANet(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 fam_head=None,
                 align_cfgs=None,
                 odm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DFTS2ANet, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            fam_head.update(train_cfg=train_cfg['fam_cfg'])
        fam_head.update(test_cfg=test_cfg)
        self.fam_head = build_head(fam_head)

        self.align_conv_type = align_cfgs['type']
        self.align_conv_size = align_cfgs['kernel_size']
        self.feat_channels = align_cfgs['channels']
        self.featmap_strides = align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if train_cfg is not None:
            odm_head.update(train_cfg=train_cfg['odm_cfg'])
        odm_head.update(test_cfg=test_cfg)
        self.odm_head = build_head(odm_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        return outs

    def unnormalize(self, x):
        # restore from T.Normalize
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # self.mean = np.array(torch.tensor(mean).view((-1, 1, 1)))
        # self.std = np.array(torch.tensor(std).view((-1, 1, 1)))
        device = x.device
        self.mean = torch.tensor(mean).view((-1, 1, 1)).to(device)
        self.std = torch.tensor(std).view((-1, 1, 1)).to(device)
        # print(x.shape)
        x = x * self.std + self.mean

        return torch.clip(x, 0, None)  # torch.clip(input, min=None, max=None) → Tensor

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        # x = self.extract_feat(img)
        ##====DFT===##
        self.patch_size = 8
        height, width = img[0].shape[-2], img[0].shape[-1]
        lpf = torch.zeros((height, width))
        R = (height + width) // self.patch_size
        for x in range(width):
            for y in range(height):
                if ((x - (width - 1) / 2) ** 2 + (y - (height - 1) / 2) ** 2) < (R ** 2):
                    # print(True)
                    lpf[y, x] = 1

        hpf = 1 - lpf
        lpf = lpf.unsqueeze(dim=-1)
        hpf = hpf.unsqueeze(dim=-1)
        img_dft = img.clone()
        imgl = img.clone()
        imgh = img.clone()
        device = img.device
        for index in range(img_dft.shape[0]):
            x = img_dft[index]
            imgx = self.unnormalize(x)
            imgx = fft.fftn(imgx, dim=(0, 1))
            imgx = torch.roll(imgx, (height // 2, width // 2),
                              dims=(0, 1))  # torch.roll(input, shifts, dims=None) → Tensor
            # print(imgx.shape, lpf.shape)
            img_l = imgx * lpf.view((lpf.shape[0], lpf.shape[1])).to(device)
            img_h = imgx * hpf.view((lpf.shape[0], lpf.shape[1])).to(device)
            imgl[index] = torch.abs(fft.ifftn(img_l, dim=(0, 1)))
            imgh[index] = torch.abs(fft.ifftn(img_h, dim=(0, 1)))
        ##=============##

        x, xl, xh = self.extract_feat(img), self.extract_feat(imgl), self.extract_feat(imgh)

        outs, outs_l, outs_h = self.fam_head(x), self.fam_head(xl), self.fam_head(xh)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_inputs_l = outs_l + (gt_bboxes, gt_labels, img_metas)
        loss_inputs_h = outs_h + (gt_bboxes, gt_labels, img_metas)

        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        loss_base_l = self.fam_head.loss(
            *loss_inputs_l, gt_bboxes_ignore=gt_bboxes_ignore)
        loss_base_h = self.fam_head.loss(
            *loss_inputs_h, gt_bboxes_ignore=gt_bboxes_ignore)

        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value
        for name, value in loss_base_l.items():
            losses[f'fam_l.{name}'] = value
        for name, value in loss_base_h.items():
            losses[f'fam_h.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        rois_l = self.fam_head.refine_bboxes(*outs_l)
        rois_h = self.fam_head.refine_bboxes(*outs_h)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        align_feat_l = self.align_conv(xl, rois_l)
        align_feat_h = self.align_conv(xh, rois_h)

        outs = self.odm_head(align_feat)
        outs_l = self.odm_head(align_feat_l)
        outs_h = self.odm_head(align_feat_h)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_inputs_l = outs_l + (gt_bboxes, gt_labels, img_metas)
        loss_inputs_h = outs_h + (gt_bboxes, gt_labels, img_metas)

        loss_refine = self.odm_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        loss_refine_l = self.odm_head.loss(
            *loss_inputs_l, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois_l)
        loss_refine_h = self.odm_head.loss(
            *loss_inputs_h, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois_h)

        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value
        for name, value in loss_refine_l.items():
            losses[f'odm_l.{name}'] = value
        for name, value in loss_refine_h.items():
            losses[f'odm_h.{name}'] = value

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
