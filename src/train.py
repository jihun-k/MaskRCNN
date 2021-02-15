import datetime
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from config import ROOT_DIR
from data import coco
from utils import box_util
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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    image_save_interval = 40
    train_image_count = 100
    max_iteration = 80000 / train_image_count
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.1)
    
    model_path = "rpn " + str(datetime.datetime.now())
    model_path = os.path.join(ROOT_DIR, "save", model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_name = "rpn.pkl"

    # model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", model_name)))

    i_mini_batch = 0
    for iter in range(max_iteration):
        print(iter, "epoch")
        i = 0
        for _, (image, annotations) in enumerate(train_loader):
            # to device
            image = image.to(device)
            
            for k, v in annotations.items():
                if torch.is_tensor(v):
                    annotations.update({k: v.to(device)})

            boxes = annotations["boxes"]
            if boxes.numel() == 0:
                continue

            # if iter==0:
            #     writer.add_image_with_boxes("GT/gt_"+str(i), image[0], box_util.xywh_to_xyxy(boxes[0]), global_step=iter)
            #     writer.flush()

            model.train()

            proposals, losses = model(image, annotations)

            print(i, losses)
            if iter != 0 and iter%image_save_interval == 0:
                writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0], global_step=iter)


            rpn_lambda = 10
            loss = losses["loss_rpn_cls"] + rpn_lambda * losses["loss_rpn_box"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i == train_image_count-1:
                writer.add_scalar("Loss/classification", losses["loss_rpn_cls"], i_mini_batch)
                writer.add_scalar("Loss/bbox_regression", losses["loss_rpn_box"], i_mini_batch)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=i_mini_batch)
                writer.flush()
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
                break
            i += 1
            i_mini_batch += 1

if __name__ == "__main__":
    main()
