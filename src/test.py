import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from config import ROOT_DIR, Config
from data import coco
from utils.transforms import ResizeSquare


def main():
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print(device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print(device)
        
    # create dataloader
    # dataset_name = "train2017"
    dataset_name = "val2017"
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", dataset_name)
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_" + dataset_name + ".json")

    train_set = coco.CocoDataset(
        imgpath,
        jsonpath, 
        device=device,
        transform=transforms.Compose([
            ResizeSquare(1024),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]),
        ]),
    )
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=coco.coco_collate)

    cfg = Config(name_prefix="test ")
    model = models.MaskRCNN(cfg)
    model.to(device)
    
    model_path = "rpn train 2021-03-04 17:07:36.358205"
    model_path = os.path.join(ROOT_DIR, "save", model_path)
    if not os.path.exists(model_path):
        print("model not exist")
        return
    model_name = "rpn.pkl"

    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    # model.eval()

    test_image_count = 110
    
    for iter in range(1):
        for i, (image, annotations) in enumerate(train_loader):

            proposals, losses = model(image, annotations)

            proposal_img0 = torch.cat([x[0] for x in proposals])
            cfg.writer.add_image_with_boxes("Image/proposal_"+str(i), image[0], proposal_img0, global_step=iter)
            cfg.writer.flush()

            if i >= test_image_count:
                break
            i += 1

if __name__ == "__main__":
    main()
