import torch
import torchvision
import os

from config import ROOT_DIR
from data import coco
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    ''' main '''
    # tensorboard writer
    writer = SummaryWriter()

    # create dataloader
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", "val2017")
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_val2017.json")

    train_set = coco.CocoDataset(imgpath, jsonpath)
    train_loader = DataLoader(train_set, batch_size=1)

    for i, (image, boxes, segments) in enumerate(train_loader):
        grid = torchvision.utils.make_grid(image)
        writer.add_image("image/show", grid, 0)
        # writer.flush()
        break
        

if __name__ == "__main__":
    main()
