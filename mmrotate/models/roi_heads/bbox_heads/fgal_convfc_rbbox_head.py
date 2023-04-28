# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .fgal_rotated_bbox_head import FGALRotatedBBoxHead

from .embedguiding import EmbedGuiding
from torch.autograd import Variable

@ROTATED_HEADS.register_module()
class FGALRotatedConvFCBBoxHead(FGALRotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(FGALRotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        divide = 4  # 2 or 4
        in_channels = int(256 / divide)
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

            self.fc_dom_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels * 7 * 7,
                out_features=self.num_dom_classes + 1)
            self.fc_sub_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels * 7 * 7,
                out_features=cls_channels)

        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

        self.softmax = nn.Softmax(dim=1)
        self.guide_high = EmbedGuiding(prior='dom')
        self.guide_low = EmbedGuiding(prior='sub')
        self.conv1 = ConvModule(256, 128, 1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.conv2 = ConvModule(256, 128, 1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def pro_coarse_score(self,coarse_score,cls_score):
        coarse_score_w = cls_score.clone()
        for i in range(coarse_score.shape[0]):
            coarse_score_w[i][0:11] = torch.full([1, 11], coarse_score[i][0].item())
            coarse_score_w[i][11:20] = torch.full([1, 9], coarse_score[i][1].item())
            coarse_score_w[i][20:30] = torch.full([1, 10], coarse_score[i][2].item())
            coarse_score_w[i][30:34] = torch.full([1, 4], coarse_score[i][3].item())
            coarse_score_w[i][34:37] = torch.full([1, 3], coarse_score[i][4].item())
            coarse_score_w[i][37] = torch.full([1, 1], coarse_score[i][5].item())
        return coarse_score_w
    def forward(self, x):
        device = x.device

        coarse_cls_score = []
        fine_cls_score = []
        fine_score_ = []
        coarse_score_ = []

        coarse_feats = x.clone()[:, :128, :, :]
        fine_feats = x.clone()[:, 128:256, :, :]

        coarse_feats = x.clone()[:, :64, :, :]
        fine_feats = x.clone()[:, 64:128, :, :]
        coarse_feats1 = x.clone()[:, 128:192, :, :]
        fine_feats1 = x.clone()[:, 192: 256, :, :]

        ## share cls head
        coarse_feats_ = coarse_feats.clone().view(coarse_feats.size(0), -1)
        coarse_score = self.fc_dom_cls(coarse_feats_)
        fine_feats_ = fine_feats.clone().view(fine_feats.size(0), -1)
        fine_score = self.fc_sub_cls(fine_feats_)

        coarse_feats1_ = coarse_feats1.clone().view(coarse_feats1.size(0), -1)
        coarse_score1 = self.fc_dom_cls(coarse_feats1_)
        fine_feats1_ = fine_feats1.clone().view(fine_feats1.size(0), -1)
        fine_score1 = self.fc_sub_cls(fine_feats1_)
        x = torch.cat((coarse_feats, fine_feats, coarse_feats1, fine_feats1), dim=1)

        coarse_feats_ = coarse_feats.clone().view(coarse_feats.size(0), -1)
        coarse_score = self.fc_dom_cls(coarse_feats_)
        coarse_score_ = Variable(coarse_score.data.clone(), requires_grad=False).to(device)
        coarse_score_ = self.softmax(coarse_score_)
        guided = self.guide_high(coarse_score_, fine_feats)
        # guided = self.guide_high(coarse_score_, coarse_feats)
        # fine_feats_pooled = torch.sum(guided.view(guided.size(0), guided.size(1), -1), dim=2)
        # fine_feats_pooled_ = fine_feats_pooled.unsqueeze(-1).unsqueeze(-1)
        feats_cat = torch.cat((coarse_feats, guided), dim=1)
        ###

        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        #
        coarse_score_w = self.pro_coarse_score(coarse_score,cls_score)
        coarse_score_w1 = self.pro_coarse_score(coarse_score1,cls_score)
        # cls_score_r = torch.mul(coarse_cls_w, cls_score)
        return coarse_score_w,fine_score,coarse_score_w1,fine_score1, cls_score, bbox_pred


@ROTATED_HEADS.register_module()
class FGALRotatedShared2FCBBoxHead(FGALRotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(FGALRotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@ROTATED_HEADS.register_module()
class FGALRotatedKFIoUShared2FCBBoxHead(FGALRotatedConvFCBBoxHead):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(FGALRotatedKFIoUShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
                bbox_targets_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode[pos_inds.type(
                        torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
