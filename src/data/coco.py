import json
import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    ''' coco dataset '''
    def __init__(
        self,
        imgpath,
        jsonpath,
        device: torch.device,
        transform=None
    ):
        ''' set path for image, caption

            Args:
                rootpath: image folder directory.
                jsonpath: annotation json file directory.
                transform: image transformer.
        '''
        super(CocoDataset, self).__init__()
        self.imgpath = imgpath
        self.jsonpath = jsonpath
        self.transform = transform
        self.device = device
        
        with open(jsonpath) as f:
            data = json.load(f)

        self.imagelist = []
        self.annotlist = []
        temp_imagelist = []
        temp_annotlist = []
        image_to_idx_dict = {}
        idx = 0
        
        for image in data["images"]:
            image_to_idx_dict[image["id"]] = idx
            temp_imagelist.append(image)
            temp_annotlist.append([])
            idx += 1

        for annotation in data["annotations"]:

            # filter only human box
            if annotation["category_id"] != 1 or annotation["iscrowd"] == 1:
                continue

            imgid = annotation["image_id"]
            if imgid not in image_to_idx_dict:
                print("wrong annotation: ", annotation)
            idx = image_to_idx_dict[annotation["image_id"]]
            temp_annotlist[idx].append(annotation)

        for i, a in zip(temp_imagelist, temp_annotlist):
            if len(a) > 0:
                self.imagelist.append(i)
                self.annotlist.append(a)
    
        self.length = len(self.imagelist)
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imagepath = os.path.join(self.imgpath, self.imagelist[idx]["file_name"])
        image = Image.open(imagepath).convert("RGB")

        # resize image
        w, h = image.size

        if self.transform is not None:
            image = self.transform(image)
            # TODO transforms.ToTensor is in self.transform?

        image = image.to(self.device)

        size = image.shape[1]

        if w >= h:
            scale = size / w
        else:
            scale = size / h

        boxes = []
        labels = []

        for item in self.annotlist[idx]:
            # if item["iscrowd"] == 0 and item["category_id"] == 1:
                labels.append(item["category_id"])
                boxes.append(item["bbox"])

        boxes = torch.tensor(boxes).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        # resize bounding box
        if len(boxes) != 0:
            # boxes[:,2] += boxes[:,0]
            # boxes[:,3] += boxes[:,1]
            boxes = boxes * scale

        annotations = {"boxes": boxes, "labels": labels}
        
        return image, annotations

    def evaluate():
        return

def coco_collate(batch):
    images = []
    annotations = []
    for image_i, annotation_i in batch:
        images.append(image_i)
        annotations.append(annotation_i)

    images = torch.stack(images)
    return images, annotations
