import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


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
        self.transform = transform
        
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

        # resize image
        w, h = image.size

        if self.transform is not None:
            image = self.transform(image)

        size = image.shape[1]

        if w >= h:
            scale = size / w
        else:
            scale = size / h

        boxes = []
        labels = []

        for item in self.annotlist[idx]:
            if item["iscrowd"] == 0 and item["category_id"] == 1:
                labels.append(item["category_id"])
                boxes.append(item["bbox"])

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        # resize bounding box
        if len(boxes) != 0:
            # boxes[:,2] += boxes[:,0]
            # boxes[:,3] += boxes[:,1]
            boxes = boxes * scale

        image_size = torch.tensor([w*scale, h*scale])

        annotations = {"boxes": boxes, "labels": labels, "image_size": image_size}
        
        return image, annotations
