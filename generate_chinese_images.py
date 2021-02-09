# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys
import random

from util import CHAR2ID_DICT, BLANK_CHAR, TRADITION_CHARS
from config import CHAR_IMG_SIZE, NUM_IMAGES_PER_FONT
from config import FONT_FILE_DIR, EXTERNEL_IMAGES_DIR
from config import CHAR_IMGS_DIR

from util import remove_then_makedirs
from img_utils import get_standard_image, get_augmented_image
from img_utils import generate_bigger_image_by_font, load_external_image_bigger

""" ************************ 矢量字体生成训练图片 *************************** """


def generate_all_chinese_images_bigger(font_file, image_size=int(CHAR_IMG_SIZE * 1.2)):
    all_chinese_list = list(CHAR2ID_DICT.keys())
    if BLANK_CHAR in all_chinese_list:
        all_chinese_list.remove(BLANK_CHAR)

    font_name = os.path.basename(font_file)
    if "繁" in font_name and "简" not in font_name:
        _chinese_chars = TRADITION_CHARS
    else:
        _chinese_chars = "".join(all_chinese_list)

    # _chinese_chars = "安愛蒼碧冒葡囊蒡夏啻坌"

    for chinese_char in _chinese_chars:
        try:  # 生成字体图片
            bigger_PIL_img = generate_bigger_image_by_font(chinese_char, font_file, image_size)
        except OSError:
            print("OSError: invalid outline, %s, %s" % (font_file, chinese_char))
            continue

        yield chinese_char, bigger_PIL_img


# 生成样例图片，以便检查图片的有效性
def generate_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=False):
    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]
    # font_file_list = [os.path.join(FONT_FINISHED_DIR, "chinese_fonts_暂时移出/康熙字典体完整版本.otf")]

    chinese_char_num = len(CHAR2ID_DICT)
    total_num = len(font_file_list) * chinese_char_num
    count = 0
    for font_file in font_file_list:  # 外层循环是字体
        font_name = os.path.basename(font_file)
        font_type = font_name.split(".")[0]

        # 创建保存该字体图片的目录
        font_img_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(font_img_dir)

        for chinese_char, bigger_PIL_img in generate_all_chinese_images_bigger(font_file, image_size=int(
                obj_size * 1.2)):  # 内层循环是字
            # 检查生成的灰度图像是否可用，黑底白字
            image_data = list(bigger_PIL_img.getdata())
            if sum(image_data) < 10:
                continue

            if not augmentation:
                PIL_img = get_standard_image(bigger_PIL_img, obj_size, reverse_color=True)
            else:
                PIL_img = get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True,
                                              reverse_color=True)

            # 保存生成的字体图片
            image_name = chinese_char + ".jpg"
            save_path = os.path.join(font_img_dir, image_name)
            PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count * 100 / total_num))
                sys.stdout.flush()
    return


def generate_chinese_images(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]

    print("Begin to generate images ...")
    chinese_char_num = len(CHAR2ID_DICT)
    total_num = len(font_file_list) * chinese_char_num
    count = 0
    for font_file in font_file_list:  # 外层循环是字体
        font_name = os.path.basename(font_file)
        font_type = font_name.split(".")[0]

        # 创建保存该字体图片的目录
        save_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(save_dir)

        for chinese_char, bigger_PIL_img in generate_all_chinese_images_bigger(font_file, image_size=int(
                obj_size * 1.2)):  # 内层循环是字
            # 检查生成的灰度图像是否可用，黑底白字
            image_data = list(bigger_PIL_img.getdata())
            if sum(image_data) < 10:
                continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True,
                                     reverse_color=True)
                 for i in range(num_imgs_per_font)]

            # 保存生成的字体图片
            for index, PIL_img in enumerate(PIL_img_list):
                image_name = chinese_char + "_" + str(index) + ".jpg"
                save_path = os.path.join(save_dir, image_name)
                PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count * 100 / total_num))
                sys.stdout.flush()


def get_external_image_paths(root_dir):
    for root, dirs, files_list in os.walk(root_dir):
        if len(files_list) > 0:
            image_paths_list = []
            for file_name in files_list:
                if file_name.endswith(".gif") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = os.path.join(root, file_name)
                    image_paths_list.append(image_path)
            if len(image_paths_list) > 0:
                font_type = os.path.basename(root)
                random.shuffle(image_paths_list)
                yield font_type, image_paths_list


# 生成样例图片，以便检查图片的有效性
def convert_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=True):
    print("Get total images num ...")
    font_images_num_list = [len(os.listdir(os.path.join(EXTERNEL_IMAGES_DIR, content)))
                            for content in os.listdir(EXTERNEL_IMAGES_DIR)
                            if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]

    print("Begin to convert images ...")
    total_num = sum(font_images_num_list)
    count = 0
    for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):
        # 创建保存该字体图片的目录
        font_img_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(font_img_dir)

        for image_path in image_paths_list:
            # 加载外部图片，将图片调整为正方形
            # 为了保证图片旋转时不丢失信息，生成的图片比本来的图片稍微bigger
            # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                print("The image %s result in OSError !" % image_path)
                continue

            if not augmentation:
                PIL_img = get_standard_image(bigger_PIL_img, obj_size, reverse_color=True)
            else:
                PIL_img = get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True,
                                              reverse_color=True)

            # 保存生成的字体图片
            image_name = os.path.basename(image_path).split(".")[0] + ".jpg"
            save_path = os.path.join(font_img_dir, image_name)
            PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count * 100 / total_num))
                sys.stdout.flush()


def convert_chinese_images(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get total images num ...")
    font_images_num_list = [len(os.listdir(os.path.join(EXTERNEL_IMAGES_DIR, content)))
                            for content in os.listdir(EXTERNEL_IMAGES_DIR)
                            if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]

    print("Begin to convert images ...")
    total_num = sum(font_images_num_list)
    count = 0
    for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):

        # 创建保存该字体图片的目录
        save_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(save_dir)

        for image_path in image_paths_list:
            # 加载外部图片，将图片调整为正方形
            # 为了保证图片旋转时不丢失信息，生成的图片比本来的图片稍微bigger
            # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                print("The image %s result in OSError !" % image_path)
                continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True,
                                     reverse_color=True)
                 for i in range(num_imgs_per_font)]

            # 保存生成的字体图片
            for index, PIL_img in enumerate(PIL_img_list):
                image_name = os.path.basename(image_path).split(".")[0] + "_" + str(index) + ".jpg"
                save_path = os.path.join(save_dir, image_name)
                PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count * 100 / total_num))
                sys.stdout.flush()


if __name__ == '__main__':
    generate_chinese_images_to_check(obj_size=200, augmentation=False)
    # generate_chinese_images(num_imgs_per_font=3)
    # convert_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=True)
    # convert_chinese_images(num_imgs_per_font=3)
