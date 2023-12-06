import numpy as np
import cv2
from PIL import Image

class BackgroundSubtractionWithSkinColor(object):
    def __init__(self, lowerb=None, upperb=None):
        self.lowerb = lowerb
        self.upperb = upperb

        if self.lowerb is None:
            self.lowerb = np.array([0, 48, 80], dtype="uint8")
        if self.upperb is None:
            self.upperb = np.array([20, 255, 255], dtype="uint8")

    def __call__(self, image: Image):
        img_array = np.array(image)
        hsv_converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        skin_mask = cv2.inRange(hsv_converted_img, self.lowerb, self.upperb)

        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        only_hand_img = cv2.bitwise_and(img_array, img_array, mask=skin_mask)

        return Image.fromarray(only_hand_img)