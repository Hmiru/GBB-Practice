import os
from PIL import Image
import random

# 사용 폴더 위치
dir_path = os.getcwd()
img_dir_path = dir_path + "/resources/images/"

folder_list = ["rock", "scissors", "paper"]

def random_rotate_img(img):
    limit = 180
    angle = random.randint(-limit, limit)

    return img.rotate(angle, expand=1)

def random_flip_img(img):
    img_direction =  [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM] #좌우, 상하
    d = img_direction[random.randint(0, 1)]

    return img.transpose(d)

def random_size_img(img):
    size = random.uniform(1,2)
    new_img = img.resize((int (img.size[0] * size), int (img.size[1]  * size)))

    return new_img

def random_move_img(img):
    limit = 800
    x_range = limit - img.size[0]
    y_range = limit - img.size[1]

    new_image = Image.new('RGB', (limit, limit), (0,0,0))
    new_image.paste(img,(random.randint(0, x_range), random.randint(0, y_range)))

    return new_image

def save_image(image_info):
    kategorie = image_info["kategorie"]
    image = image_info["img"]
    name = image_info["name"]

    image.save(img_dir_path + "transform/" + kategorie + '/' + name + ".png", "JPEG")
def create_transform_image():
    for gbb in folder_list:
        gbb_folder_path = img_dir_path + gbb
        gbb_file_list = os.listdir(gbb_folder_path)
        for idx in range(3000):
            img = Image.open(gbb_folder_path + "/" + gbb_file_list[random.randrange(0, len(gbb_file_list))])
            img = random_rotate_img(img)
            img = random_flip_img(img)
            img = random_size_img(img)
            img = random_move_img(img)

            save_image({"kategorie":gbb, "img": img, "name": gbb + str(idx)})

create_transform_image()