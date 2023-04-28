# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage_dft import RotatedSingleStageDetectorDFT


@ROTATED_DETECTORS.register_module()
class RotatedRepPointsDFT(RotatedSingleStageDetectorDFT):
    """Implementation of Rotated RepPoints."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RotatedRepPointsDFT, self).__init__(backbone, neck, bbox_head,
                                               train_cfg, test_cfg, pretrained)
