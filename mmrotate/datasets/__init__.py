# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .fair1m import FAIR1MDataset
from .fg_fair1m import FGFAIR1MDataset
from .mar20 import MARDataset

__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset','FAIR1MDataset',
           'FGFAIR1MDataset',"MARDataset"]
