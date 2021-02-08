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
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", "val2017")
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_val2017.json")

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

    max_iteration = 1000
    
    for epoch in range(max_iteration):
        print(epoch, "epoch")
        i = 0
        for _, (image, boxes, labels) in enumerate(train_loader):
            if boxes.numel() == 0:
                continue

            image, boxes = image.to(device), boxes.to(device)

            model.train()

            proposals, losses = model(image, boxes)

            print(i, losses)
            if epoch == max_iteration-1 and i < 100:
                writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0], global_step=epoch)

            if i == 0:
                if epoch%10 == 0:
                    writer.add_scalar("Loss/classification", losses["loss_rpn_cls"], epoch)
                    writer.add_scalar("Loss/bbox_regression", losses["loss_rpn_box"], epoch)
                    writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0], global_step=epoch)
                    writer.flush()
                    model_name = "rpn.pkl"
                    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", model_name))
                break

            rpn_lambda = 10
            losses = losses["loss_rpn_cls"] + rpn_lambda * losses["loss_rpn_box"]

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1

if __name__ == "__main__":
    main()
