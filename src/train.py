import torch
import torchvision
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import ROOT_DIR
from data import coco
import models

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

    train_set = coco.CocoDataset(imgpath, jsonpath)
    train_loader = DataLoader(train_set, batch_size=1)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = models.build_model()
    model.to(device)

    for i, (image, annotations) in enumerate(train_loader):
        '''
        # draw ground truth bounding boxes on image
        box_clone = boxes.detach().clone()
        box_clone[0,:,2] += box_clone[0,:,0]
        box_clone[0,:,3] += box_clone[0,:,1]
        writer.add_image_with_boxes("image/debug", image[0], box_clone[0])
        writer.flush()
        break
        '''
        model.train(False)
        res = model(image)

        print(res.shape)

        '''
        # draw proposed bounding boxes on image
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

        break

if __name__ == "__main__":
    main()
