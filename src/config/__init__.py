import datetime
import os

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
        self.backbone_init_imagenet = False
        self.batch_size = 1
        self.freeze_backbone = freeze_backbone
        self.freeze_rpn = freeze_rpn
        return

