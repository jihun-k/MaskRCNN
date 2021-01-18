import PIL
import torch
import json
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image

class CocoDataset(Dataset):
    ''' coco dataset '''
    def __init__(self, imgpath, jsonpath, transform=None):
        ''' set path for image, caption

            Args:
                rootpath: image folder directory.
                jsonpath: annotation json file directory.
                transform: image transformer.
        '''
        self.imgpath = imgpath
        self.jsonpath = jsonpath
        
        with open(jsonpath) as f:
            data = json.load(f)

        self.length = len(data["images"])
        
        self.imagelist = []
        self.annotlist = []
        image_to_idx_dict = {}
        idx = 0
        
        for image in data["images"]:
            image_to_idx_dict[image["id"]] = idx
            self.imagelist.append(image)
            self.annotlist.append([])
            idx += 1

        for annotation in data["annotations"]:
            imgid = annotation["image_id"]
            if imgid not in image_to_idx_dict:
                print("wrong annotation: ", annotation)
            idx = image_to_idx_dict[annotation["image_id"]]
            self.annotlist[idx].append(annotation)

        # print(self.annotlist[0])
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imagepath = os.path.join(self.imgpath, self.imagelist[idx]["file_name"])
        image = Image.open(imagepath).convert("RGB")
        image = transforms.ToTensor()(image)

        boxes = []
        labels = []

        for item in self.annotlist[idx]:
            labels.append(item["category_id"])
            boxes.append(item["bbox"])

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        annotations = {}
        annotations["boxes"] = boxes
        annotations["labels"] = labels
        
        return image, annotations
