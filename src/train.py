from config import ROOT_DIR
import torch
import os

from data import coco
from torch.utils.tensorboard import SummaryWriter

def main():
    ''' main '''
    # create dataloader
    imgpath = os.path.join(ROOT_DIR, "datasets", "COCO", "train2017")
    jsonpath = os.path.join(ROOT_DIR, "datasets", "COCO", "annotations", "instances_train2017.json")
    dataset = coco.CocoDataset(imgpath, jsonpath)

    '''
    image, boxes, segments = dataset[0]
    image_tensor = torch.tensor(image)
    print(image_tensor.size())
    print(boxes)
    print(segments)
    '''

    '''
    writer = SummaryWriter()

    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    def train_model(iter):
        for epoch in range(iter):
            y1 = model(x)
            loss = criterion(y1, y)
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_model(1000)
    writer.flush()
    '''

if __name__ == "__main__":
    main()
