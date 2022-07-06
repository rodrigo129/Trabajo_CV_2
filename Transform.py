import torch
from torchvision import transforms
import torchvision
from torchvision.transforms import InterpolationMode


def GetTransform(test = False):
    """
    function that give a transform function for images

    :return:
    """
    if test:
        return transforms.Compose([torchvision.transforms.RandomRotation(60, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None), torchvision.transforms.RandomVerticalFlip(p=0.5), torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0)), torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0)), transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.3), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
