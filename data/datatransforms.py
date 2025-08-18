import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from PIL.Image import Transpose
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        img_B = np.array(img_B).astype(np.float32).transpose((2, 0, 1))
        label_BCD = np.array(label_BCD).astype(np.float32)
        label_SGA = np.array(label_SGA).astype(np.float32)
        label_SGB = np.array(label_SGB).astype(np.float32)

        img_A = torch.from_numpy(img_A).type(torch.FloatTensor)
        img_B = torch.from_numpy(img_B).type(torch.FloatTensor)
        label_BCD = torch.from_numpy(label_BCD).type(torch.LongTensor)
        label_SGA = torch.from_numpy(label_SGA).type(torch.LongTensor)
        label_SGB = torch.from_numpy(label_SGB).type(torch.LongTensor)
        
        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            label_BCD = label_BCD.transpose(Image.FLIP_LEFT_RIGHT)
            label_SGA = label_SGA.transpose(Image.FLIP_LEFT_RIGHT)
            label_SGB = label_SGB.transpose(Image.FLIP_LEFT_RIGHT)

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_TOP_BOTTOM)
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)
            label_BCD = label_BCD.transpose(Image.FLIP_TOP_BOTTOM)
            label_SGA = label_SGA.transpose(Image.FLIP_TOP_BOTTOM)
            label_SGB = label_SGB.transpose(Image.FLIP_TOP_BOTTOM)

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Transpose.ROTATE_90, Transpose.ROTATE_180, Transpose.ROTATE_270]

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            rotate_degree = random.choice(self.degree)
            img_A = img_A.transpose(rotate_degree)
            img_B = img_B.transpose(rotate_degree)
            label_BCD = label_BCD.transpose(rotate_degree)
            label_SGA = label_SGA.transpose(rotate_degree)
            label_SGB = label_SGB.transpose(rotate_degree)            

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomRotate(object):
    def __init__(self, degree, fillcolor=0):
        self.degree = degree
        self.fillcolor = fillcolor

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img_A = img_A.rotate(rotate_degree, Image.BILINEAR, expand=True)
            img_B = img_B.rotate(rotate_degree, Image.BILINEAR, expand=True)
            label_BCD = label_BCD.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=255)
            label_SGA = label_SGA.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=self.fillcolor)
            label_SGB = label_SGB.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=self.fillcolor)     

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        assert img_A.size == label_BCD.size and img_B.size == label_BCD.size == label_SGA.size == label_SGB.size

        img_A = img_A.resize(self.size, Image.BILINEAR)
        img_B = img_B.resize(self.size, Image.BILINEAR)
        label_BCD = label_BCD.resize(self.size, Image.NEAREST)
        label_SGA = label_SGA.resize(self.size, Image.NEAREST)
        label_SGB = label_SGB.resize(self.size, Image.NEAREST)    

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class ColorJitterImages(object):
    def __init__(self):
        self.colorjitter = transforms.Compose([transforms.ColorJitter(0.2, 0.4, 0.4, 0.1)])

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        f = random.random() 
        if (f > 0.25) and (f <= 0.5):
            img_A = self.colorjitter(img_A)
            img_B = self.colorjitter(img_B)
        elif (f > 0.5) and (f <= 0.75):
            img_A = self.colorjitter(img_A)
        elif (f > 0.75) and (f <= 1):
            img_B = self.colorjitter(img_B)
        
        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }
        

class GaussianBlur(object):
    """
    高斯模糊
    """
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        f = random.random() 
        if (f > 0.25) and (f <= 0.5):
            img_A = img_A.filter(ImageFilter.GaussianBlur(radius=1))
        elif (f > 0.5) and (f <= 0.75):
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=1))
        elif (f > 0.75) and (f <= 1):
            img_A = img_A.filter(ImageFilter.GaussianBlur(radius=1))
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=1))
        
        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }
        

test_transforms = transforms.Compose([ToTensor()])


def get_train_transforms(with_colorjit=False, rotate=True):
    steps = [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ]
    if rotate:
        steps.append(RandomFixRotate())
    if with_colorjit:
        steps.append(ColorJitterImages())
    steps.append(ToTensor())
    
    train_transforms = transforms.Compose(steps)        
    return train_transforms


if  __name__ == '__main__':
    
    pass