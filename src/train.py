import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from config import ROOT_DIR
from data import coco
from utils.transforms import ResizeSquare


def main():
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print(device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print(device)
        
    # tensorboard writer
    writer = SummaryWriter()

    # create dataloader
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", "train2017")
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_train2017.json")

    train_set = coco.CocoDataset(imgpath,
                                 jsonpath, 
                                 transforms.Compose([
                                    ResizeSquare(1024),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #     std=[0.229, 0.224, 0.225]),
                                ]))
    train_loader = DataLoader(train_set, batch_size=1)

    model = models.MaskRCNN(writer)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.rpn.parameters(), lr=0.01, momentum=0.9)
    # TODO lr scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    model.train()
    
    for epoch in range(1000):
        print(epoch, "epoch")
        for i, (image, boxes, labels) in enumerate(train_loader):

            image, boxes = image.to(device), boxes.to(device)

            model.train()

            proposals, losses = model(image, boxes)

            print(i, losses)
            if i % 1000 == 0:
                writer.add_scalar("Loss/classification", losses["loss_rpn_cls"], epoch)
                writer.add_scalar("Loss/bbox_regression", losses["loss_rpn_box"], epoch)
                writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0])
                writer.flush()
                model_name = "rpn_" + str(epoch) + "_" + str(i) + ".pkl"
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", model_name))
                

            rpn_lambda = 10
            losses = losses["loss_rpn_cls"] + rpn_lambda * losses["loss_rpn_box"]

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
