import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from config import ROOT_DIR
from data import coco
from utils.transforms import ResizeSquare


def main():
    ''' main '''
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        # print(device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        # print(device)
        
    # tensorboard writer
    writer = SummaryWriter()

    # create dataloader
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", "val2017")
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_val2017.json")

    train_set = coco.CocoDataset(imgpath,
                                 jsonpath, 
                                 transforms.Compose([
                                    ResizeSquare(1024),
                                    transforms.ToTensor()
                                ]))
    train_loader = DataLoader(train_set, batch_size=1)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # print(model)
    model = models.build_model()
    model.to(device)

    for i, (image, annotations) in enumerate(train_loader):
        
        image, boxes, labels = image.to(device), annotations["boxes"].to(device), annotations["labels"].to(device)
        model.train(False)
        # res = model(image, annotations)

        '''
        # draw ground truth bounding boxes on image
        if i == 6:
            writer.add_image_with_boxes("image/debug", image[0], boxes[0])
            writer.flush()
            break
        '''

        '''
        # draw result bounding boxes on image
        for batch in range(len(image)):
            count = len(res[0]["boxes"])
            boxes = res[batch]["boxes"]
            labels = res[batch]["labels"]
            scores = res[batch]["scores"]
            boxes_valid = []
            for i in range(count):
                if scores[i] < 0.5:
                    continue
                else:
                    boxes_valid.append(boxes[i])
            writer.add_image_with_boxes("image/debug", image[batch], res[batch]["boxes"])
        '''

if __name__ == "__main__":
    main()
