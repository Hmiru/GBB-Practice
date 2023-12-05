import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import yaml
from torchvision.transforms import Compose, Grayscale, Resize, Pad, ToTensor, Lambda
import torch
import matplotlib.pyplot as plt
import os
import cv2

import numpy as np
from PIL import Image
import yaml
with open('config.yaml') as f:
    file = yaml.full_load(f)
def hand_detection_with_skincolor(bgr_image, lowerb=None, upperb=None):
    if lowerb is None:
        lowerb = np.array([0, 48, 80], dtype="uint8")
    if upperb is None:
        upperb = np.array([20, 255, 255], dtype="uint8")

    hsv_converted_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_converted_img, lowerb, upperb)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    only_hand_img = cv2.bitwise_and(bgr_image, bgr_image, mask=skin_mask)

    return only_hand_img


def skincolor_subtraction(bgr_image):
    converted_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    return converted_img

class BackgroundSubtractionWithSkinColor(object):
    def __init__(self, lowerb=None, upperb=None):
        self.lowerb = lowerb
        self.upperb = upperb

        if self.lowerb is None:
            self.lowerb = np.array([0, 48, 80], dtype="uint8")
        if self.upperb is None:
            self.upperb = np.array([20, 255, 255], dtype="uint8")

    def __call__(self, image):
        img_array = np.array(image)
        hsv_converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        skin_mask = cv2.inRange(hsv_converted_img, self.lowerb, self.upperb)

        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        only_hand_img = cv2.bitwise_and(img_array, img_array, mask=skin_mask)

        return Image.fromarray(only_hand_img)


# 정사각형으로 이미지를 확장시키기
class ExpandRectangle(object):
    def __call__(self, image):
        original_width, original_height = image.size

        if original_width == original_height:
            return image
        elif original_width > original_height:
            diff = original_width - original_height
            pad_tuple = (0, diff // 2) if diff % 2 == 0 else (0, diff // 2, 0, diff // 2 + 1)
        else:
            diff = original_height - original_width
            pad_tuple = (diff // 2, 0) if diff % 2 == 0 else (diff // 2, 0, diff // 2 + 1, 0)

        return Pad(pad_tuple)(image)


# 전처리 과정을 조합한 하나의 Transform 만들기 (size로 크기의 정사각형을 만드는)
def make_composed_transform_with_size(size):
    return Compose([
        BackgroundSubtractionWithSkinColor(),
        ExpandRectangle(),
        Resize(size),
        Grayscale(1),
        ToTensor()
    ])

def make_one_hot_transform_with_class_num(num):
    return Lambda(lambda label: torch.eye(num)[label])

if __name__=="__main__":
    DATASET_LOCATION = file["dataset_path"]
    EXAMPLE_IMAGE = cv2.imread(DATASET_LOCATION + "merged_example.png")
    REAL_IMAGE = cv2.imread(DATASET_LOCATION + "real_image/1.jpg")

    example_bg_subtraction_image = hand_detection_with_skincolor(EXAMPLE_IMAGE)
    example_skincolor_subtraction_image = skincolor_subtraction(example_bg_subtraction_image)


    plt.imshow(cv2.cvtColor(example_skincolor_subtraction_image, cv2.COLOR_BGR2RGB))
    plt.show()


