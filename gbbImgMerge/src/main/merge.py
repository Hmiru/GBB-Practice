import os
from PIL import Image
import random

# 사용 폴더 위치
dir_path = os.getcwd()
img_dir_path = dir_path + "/resources/images/"

folder_list = ["rock", "scissors", "paper"]

def random_img_rotate(img):
    limit = 70
    angle = random.randint(-limit, limit)

    return img.rotate(angle, expand=1)

def paste_image(img1, img2):
    len = 361
    lenlen = len + len

    new_image = Image.new('RGB', (lenlen, lenlen), (0,0,0))
    new_image.paste(img1,(0, random.randint(20, len - 20)))
    new_image.paste(img2,(len, random.randint(20, len - 20)))

    return new_image

def merge_image(img_path1, img_path2):
    image1 =Image.open(img_path1)
    image2 = Image.open(img_path2)

    image1 = random_img_rotate(image1.rotate(180,expand=1))
    image2 = random_img_rotate(image2)

    return paste_image(image1, image2)

def save_image(image_info):
    kategorie = image_info["kategorie"]
    image = image_info["img"]
    name = image_info["name"]

    image.save(img_dir_path + "merged/" + kategorie + '/' + name + ".png", "JPEG")

def create_merge_image():
    for left in folder_list:
        left_folder_path = img_dir_path + left
        left_file_list = os.listdir(left_folder_path)
        for right in folder_list:
            kategorie = left + '_' + right
            right_foler_path = img_dir_path + right
            right_file_list = os.listdir(right_foler_path)

            for idx in range(1000):
                image1_path = left_folder_path + "/" + left_file_list[random.randrange(0, len(left_file_list))]
                image2_path = right_foler_path + "/" + right_file_list[random.randrange(0, len(right_file_list))]

                mergeImg = merge_image(image1_path, image2_path)
                save_image({"kategorie":kategorie, "img": mergeImg, "name": kategorie + str(idx)})

create_merge_image()