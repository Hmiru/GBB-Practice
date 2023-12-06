from .background_subtraction import BackgroundSubtractionWithSkinColor
from .expand_square import ExpandSquare
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, Lambda
from torch import eye

def make_composed_transform_with_size(size):
    return Compose([
        BackgroundSubtractionWithSkinColor(),
        ExpandSquare(),
        Resize(size),
        Grayscale(1),
        ToTensor()
    ])

def make_one_hot_transform_with_class_num(num):
    return Lambda(lambda label: eye(num)[label])