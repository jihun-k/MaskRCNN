import datetime
import os
from typing import Tuple

from torch.utils.tensorboard.writer import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config():
    def __init__(
        self,
        name_prefix="",
        freeze_backbone=False,
        freeze_rpn=False,
    ):
        self.name = name_prefix + str(datetime.datetime.now())
        self.writer = SummaryWriter(os.path.join("runs", self.name))
        self.backbone_init_imagenet = True
        self.batch_size = 1
        self.freeze_backbone = freeze_backbone
        self.freeze_rpn = freeze_rpn
        self.image_size = [1024, 1024]
        return

