from img_utils import reverse_image_color
from ocrodeg import *
from PIL import Image, ImageDraw
import numpy as np

def change_seal_color(PIL_page):
    background_color_r = random.randint(230, 235)
    background_color_g = random.randint(230, 235)
    background_color_b = random.randint(180, 235)

    seal_color_r = random.randint(200, 235)
    seal_color_g = random.randint(10, 120)
    seal_color_b = seal_color_g + random.randint(-10, 10)

    gray_page = np.array(PIL_page, dtype=np.uint8)

    # 创建新图像容器
    rgb_page = np.zeros(shape=(*gray_page.shape, 3))
    # 遍历每个像素点
    for i in range(rgb_page.shape[0]):
        for j in range(rgb_page.shape[1]):
            offset_r = random.randint(0, 20)
            offset_g = random.randint(0, 20)
            offset_b = random.randint(0, 20)

            chromatism_r = background_color_r - seal_color_r
            chromatism_g = background_color_g - seal_color_g
            chromatism_b = background_color_r - seal_color_b

            if gray_page[i, j] > 150:
                color_bg = [gray_page[i, j] + background_color_r + offset_r - 255,
                            gray_page[i, j] + background_color_g + offset_g - 255,
                            gray_page[i, j] + background_color_b + offset_b - 255]
                rgb_page[i, j, :] = color_bg
            else:
                color_seal = [seal_color_r + int(gray_page[i, j] / 255 * chromatism_r) + offset_r,
                              seal_color_g + int(gray_page[i, j] / 255 * chromatism_g) + offset_g,
                              seal_color_b + int(gray_page[i, j] / 255 * chromatism_b) + offset_b]
                rgb_page[i, j, :] = color_seal
    PIL_rgb_page = Image.fromarray(rgb_page.astype(np.uint8))
    return PIL_rgb_page
    # return rgb_page.astype(np.uint8)

def change_seal_shape(PIL_page, text_bbox_list, char_bbox_list):
    np_seal = np.array(PIL_page, dtype=np.uint8)
    height_seal, width_seal = np_seal.shape

    margin = min(round(random.uniform(0.05, 0.1) * width_seal),
                 round(random.uniform(0.05, 0.1) * height_seal))

    height_bg = height_seal + 4 * margin
    width_bg = width_seal + 4 * margin
    np_background = np.zeros(shape=(height_bg, width_bg), dtype=np.uint8)

    # 确定印章坐标
    x1_seal = 2 * margin
    x2_seal = x1_seal + width_seal
    y1_seal = 2 * margin
    y2_seal = y1_seal + height_seal

    # 一半的概率阴阳印章
    if random.random() < 0.5:
        np_seal = reverse_image_color(np_img=np_seal)
    PIL_page = Image.fromarray(np_seal)

    np_page, text_bbox_list, char_bbox_list = add_subpage_into_page(
        np_background, PIL_page, text_bbox_list, char_bbox_list, x1_seal, x2_seal, y1_seal, y2_seal
    )

    if random.random() < 0.5:  # 一半概率加边框
        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

        rect_line_width = int(margin / 2)
        draw.rectangle([(x1_seal - margin, y1_seal - margin), (x2_seal + margin, y2_seal + margin)],
                       fill=None, outline="white", width=rect_line_width)
        np_page = np.array(PIL_page, dtype=np.uint8)

    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    return PIL_page, text_bbox_list, char_bbox_list

def add_subpage_into_page(np_page, PIL_subpage, text_bbox_list, char_bbox_list, x1, x2, y1, y2, cover=False):
    np_subpage = np.array(PIL_subpage, dtype=np.uint8)
    np_subpage = reverse_image_color(np_img=np_subpage)
    if cover:  # 完全覆盖
        np_page[y1:y2, x1:x2] = np_subpage
    else:
        np_page[y1:y2, x1:x2] |= np_subpage

    bbox_list_list = [text_bbox_list, char_bbox_list]

    for bbox_list in bbox_list_list:
        for bbox in bbox_list:
            bbox[0] += x1
            bbox[1] += y1
            bbox[2] += x1
            bbox[3] += y1

    return np_page, text_bbox_list, char_bbox_list