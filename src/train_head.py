import datetime
import os

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

    cfg = Config(name_prefix="train ")
    cfg.backbone_init_imagenet = True

    
    train_image_count = 60000
    max_iteration = int(80000 / train_image_count)
    max_iteration = 2
    image_save_count = 100

    model = models.MaskRCNN(cfg)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.1)

    model_path = "rpn 2021-02-15 14:38:12.256264"
    model_path = os.path.join(ROOT_DIR, "save", model_path)
    if not os.path.exists(model_path):
        print("model not exist")
        return
    model_name = "rpn.pkl"

    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", model_name)))

    i_mini_batch = 0
    for iter in range(max_iteration):
        print(iter, "epoch")
        i = 0
        for _, (image, annotations) in enumerate(train_loader):
            image = image.to(device)
            
            for k, v in annotations.items():
                if torch.is_tensor(v):
                    annotations.update({k: v.to(device)})

            boxes = annotations["boxes"]
            if boxes.numel() == 0:
                continue

            model.train()

            proposals, losses = model(image, annotations)

            print(i, losses)
            # if iter != 0 and iter%image_save_interval == 0 or iter == max_iteration-1:
            if i % (train_image_count / image_save_count) == 0:
                    cfg.writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposals[0], global_step=iter)


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
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            i += 1
            if i == train_image_count:
                break
    
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

if __name__ == "__main__":
    main()
