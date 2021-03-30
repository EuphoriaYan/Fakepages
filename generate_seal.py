from img_utils import reverse_image_color, find_min_bound_box
from ocrodeg import *
from PIL import Image, ImageDraw
import numpy as np
import cv2

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
        shadow_seal = True
    else:
        shadow_seal = False

    PIL_page = Image.fromarray(np_seal)

    outline_width = int(margin / 2)
    np_seal, np_outline = round_corner(np_seal, outline_width, outline=True)
    if random.random() < 0.5 or not shadow_seal:  # 一半概率加边框，若是阳刻印章，则加上边框
        if not shadow_seal:
            np_outline = resize_img_by_opencv(np_outline, obj_size=(width_seal, height_seal))
        h_outline, w_outline = np_outline.shape
        w_margin = int((width_bg - w_outline)/2)
        h_margin = int((height_bg - h_outline)/2)
        np_background[h_margin:h_margin + h_outline, w_margin:w_margin + w_outline] |= np_outline

    np_page, text_bbox_list, char_bbox_list = add_subpage_into_page(
        np_background, PIL_page, text_bbox_list, char_bbox_list, x1_seal, x2_seal, y1_seal, y2_seal
    )
        # PIL_page = Image.fromarray(np_page)
        # draw = ImageDraw.Draw(PIL_page)
        #
        # rect_line_width = int(margin / 2)
        # draw.rectangle([(x1_seal - margin, y1_seal - margin), (x2_seal + margin, y2_seal + margin)],
        #                fill=None, outline="white", width=rect_line_width)
        # np_page = np.array(PIL_page, dtype=np.uint8)

    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    return PIL_page, text_bbox_list, char_bbox_list

def round_corner(np_seal, line_width, outline=False):
    circle_d = min(np_seal.shape[0], np_seal.shape[1])  # 圆的直径

    # 画实心圆，用于遮挡印章
    np_circle = np.zeros(shape=(circle_d, circle_d), dtype=np.uint8)
    PIL_circle = Image.fromarray(np_circle)
    draw = ImageDraw.Draw(PIL_circle)
    draw.ellipse((0, 0, PIL_circle.size[0], PIL_circle.size[1]), fill="white", outline="white", width=line_width)
    np_circle = np.array(PIL_circle, dtype=np.uint8)

    # 画空心圆，用于边框
    np_corner = np.zeros(shape=(circle_d, circle_d), dtype=np.uint8)
    PIL_corner = Image.fromarray(np_corner)
    draw = ImageDraw.Draw(PIL_corner)
    draw.ellipse((0, 0, PIL_corner.size[0], PIL_corner.size[1]), fill=None, outline="white", width=line_width)
    np_corner = np.array(PIL_corner, dtype=np.uint8)

    np_outline = np.zeros(shape=(np_seal.shape[0] + line_width*4, np_seal.shape[1] + line_width*4), dtype=np.uint8)
    PIL_outline = Image.fromarray(np_outline)
    draw = ImageDraw.Draw(PIL_outline)
    draw.rectangle((0, 0, PIL_outline.size[0], PIL_outline.size[1]), fill=None, outline="white", width=line_width)
    np_outline = np.array(PIL_outline, dtype=np.uint8)

    # 切出四个弧度角
    for x in (0, circle_d):
        for y in (0, circle_d):
            corner_x = random.randint(int(circle_d*0.15), int(circle_d*0.2))
            corner_y = random.randint(int(circle_d*0.15), int(circle_d*0.2))
            x1_in_seal = 1
            y1_in_seal = 1
            if x == 0:
                x1 = x
                x2 = x + corner_x
                x1_in_seal = 0
            else:
                x1 = x - corner_x
                x2 = x
            if y == 0:
                y1 = y
                y2 = y + corner_y
                y1_in_seal = 0
            else:
                y1 = y - corner_y
                y2 = y

            np_corner_circle = np_circle[y1:y2, x1:x2]
            left, right, top, low = find_min_bound_box(np_corner_circle)
            np_corner_circle = np_corner_circle[top:low + 1, left:right + 1]
            np_corner_circle = reverse_image_color(np_corner_circle)  # 留下的部分变为白色，即圆弧以外的部分

            if x1_in_seal != 0:
                x1_in_seal = np_seal.shape[1] - np_corner_circle.shape[1]
            if y1_in_seal != 0:
                y1_in_seal = np_seal.shape[0] - np_corner_circle.shape[0]

            np_seal[y1_in_seal:y1_in_seal+np_corner_circle.shape[0], x1_in_seal:x1_in_seal+np_corner_circle.shape[1]] |= np_corner_circle

            if outline:
                np_corner_line = np_corner[y1:y2, x1:x2]
                left, right, top, low = find_min_bound_box(np_corner_line)
                if x1_in_seal != 0:
                    x1_in_seal += line_width * 4
                if y1_in_seal != 0:
                    y1_in_seal += line_width * 4
                np_corner_line = np_corner_line[top:low + 1, left:right + 1]
                np_outline[y1_in_seal:y1_in_seal+np_corner_circle.shape[0],
                    x1_in_seal:x1_in_seal+np_corner_circle.shape[1]] = np_corner_line

    return np_seal, np_outline

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

def resize_img_by_opencv(np_img, obj_size, make_border=False):
    cur_height, cur_width = np_img.shape[:2]
    obj_width, obj_height = obj_size

    # cv.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # dsize为目标图像大小，(fx, fy)为(横, 纵)方向的缩放比例，参数dsize和参数(fx, fy)不必同时传值
    # interpolation为插值方法，共有5种：INTER_NEAREST 最近邻插值法，INTER_LINEAR 双线性插值法(默认)，
    # INTER_AREA 基于局部像素的重采样，INTER_CUBIC 基于4x4像素邻域的3次插值法，INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值
    # 如果是缩小图片，效果最好的是INTER_AREA；如果是放大图片，效果最好的是INTER_CUBIC(slow)或INTER_LINEAR(faster but still looks OK)
    if obj_height == cur_height and obj_width == cur_width:
        return np_img
    elif obj_height + obj_width < cur_height + cur_width:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    if make_border:
        border_type = cv2.BORDER_CONSTANT
        np_img = cv2.copyMakeBorder(np_img, 0, 0, cur_width // 2, 0, border_type, value=0)

    resized_np_img = cv2.resize(np_img, dsize=(obj_width, obj_height), interpolation=interpolation)

    return resized_np_img