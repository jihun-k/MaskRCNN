from PIL import Image
from torch import tensor
from torchvision import transforms
from torchvision.transforms.functional import resize
import numpy as np

class ResizeSquare(object):
    '''
        resize image to given size. longer edge of image will be matched to the number, then padding will be added
        
    '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if w == self.size and h == self.size:
            return img
        if w == h:
            return img.resize((self.size, self.size), self.interpolation)
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            pad_top = 0 #int((ow - oh) / 2)
            pad_bottom = ow - oh #- pad_top
            img = img.resize((ow, oh), self.interpolation)
            img = np.array(img)
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)))
            return Image.fromarray(img)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            pad_left = 0 #int((oh - ow) / 2)
            pad_right = oh - ow #- pad_left
            img = img.resize((ow, oh), self.interpolation)
            img = np.array(img)
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)))
            return Image.fromarray(img)


