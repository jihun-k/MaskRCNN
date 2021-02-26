import datetime
import os

from torch.utils.tensorboard.writer import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config():
    def __init__(self, name_prefix=""):
        self.name = name_prefix + str(datetime.datetime.now())
        self.writer = SummaryWriter(os.path.join("runs", self.name))
        self.backbone_init_imagenet = False
        return

