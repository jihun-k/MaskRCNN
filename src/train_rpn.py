import datetime
import os
import shutil
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from config import ROOT_DIR, Config
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

    cfg = Config(name_prefix="rpn train ")
    cfg.backbone_init_imagenet = True
    cfg.batch_size = 1
    cfg.image_size = [1024, 1024]
        
    # dataset_name = "train2017"
    dataset_name = "val2017"
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", dataset_name)
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_" + dataset_name + ".json")
    train_set = coco.CocoDataset(
        imgpath,
        jsonpath,
        device=device,
        transform=transforms.Compose([
        ResizeSquare(cfg.image_size[0]),
        transforms.ToTensor(),
    ]))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, collate_fn=coco.coco_collate)

    # train_image_count = len(train_set)
    train_image_count = 20000
    max_iteration = 4
    image_save_count = 100
    image_save_interval = int(train_image_count / image_save_count)
    image_save_iter_interval = 1
    rpn_proposal_draw_count = min(100, 1000) # num_boxes, 1000

    model = models.MaskRCNN(cfg)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60000], gamma=0.1)
    
    model_path = os.path.join(ROOT_DIR, "save", cfg.name)
    model_name = "rpn.pkl"

    # model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", model_name)))
    i_mini_batch = 0
    max_i_mini_batch = 80000
    for iter in range(max_iteration):
        print(iter, "epoch")
        for i, (image, annotations) in enumerate(train_loader):

            model.train()

            proposals, losses = model(image, annotations)

            print(i, losses)
            if i % image_save_interval == 0:
                if iter % image_save_iter_interval == 0:
                    cfg.writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0][:rpn_proposal_draw_count-1], global_step=iter)
                # if iter == 0:
                #     # gt
                #     cfg.writer.add_image_with_boxes("GT/gt_"+str(i), image[0], box_util.xywh_to_xyxy(annotations[0]["boxes"]))


            rpn_lambda = 10
            loss = losses["loss_rpn_cls"] + rpn_lambda * losses["loss_rpn_box"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            cfg.writer.add_scalar("Loss/classification", losses["loss_rpn_cls"], i_mini_batch)
            cfg.writer.add_scalar("Loss/bbox_regression", losses["loss_rpn_box"], i_mini_batch)
            cfg.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=i_mini_batch)
            cfg.writer.flush()

            i_mini_batch += 1
            if i_mini_batch % 1000 == 0:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model.state_dict(), os.path.join(model_path, model_name))

            if i == train_image_count-1:
                break

            if i_mini_batch == max_i_mini_batch-1:
                break
        if i_mini_batch == max_i_mini_batch-1:
            break
    
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

if __name__ == "__main__":
    main()
