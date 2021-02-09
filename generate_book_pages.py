# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import json
import random
import numpy as np
from PIL import Image, ImageDraw
from queue import Queue

from util import check_or_makedirs
from img_utils import reverse_image_color
from generate_text_lines import check_text_type
from generate_text_lines import generate_mix_rows_chars, generate_mix_cols_chars

from config import BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
from config import BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
from config import BOOK_PAGE_SHAPE_LIST


def generate_book_page_imgs(obj_num=10, text=None, text_type="horizontal", init_num=0, page_shape=None):
    text_type = check_text_type(text_type)

    if text_type == "h":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
    elif text_type == "v":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
    else:
        raise ValueError('text_type should be horizontal or vertical')

    check_or_makedirs(book_page_imgs_dir)

    _shape = page_shape
    with open(book_page_tags_file, "w", encoding="utf-8") as fw:
        for i in range(init_num, init_num + obj_num):
            '''
            if page_shape is None and text_type == "h":
                _shape = (random.randint(480, 720), random.randint(640, 960))
            if page_shape is None and text_type == "v":
                _shape = (random.randint(640, 960), random.randint(480, 720))
            '''
            if page_shape is None:
                _shape = random.choice(BOOK_PAGE_SHAPE_LIST)

            PIL_page, text_bbox_list, split_pos_list = create_book_page(_shape, text=text, text_type=text_type)
            image_tags = {"text_bbox_list": text_bbox_list, "split_pos_list": split_pos_list}

            img_name = "book_page_%d.jpg" % i
            save_path = os.path.join(book_page_imgs_dir, img_name)
            PIL_page.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")

            if i % 50 == 0:
                print(" %d / %d Done" % (i, obj_num))
                sys.stdout.flush()


def generate_book_page_imgs_with_img(obj_num=10, text_type="horizontal", init_num=0, page_shape=None):
    text_type = check_text_type(text_type)

    if text_type == "h":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
    elif text_type == "v":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
    else:
        raise ValueError('text_type should be horizontal or vertical')

    check_or_makedirs(book_page_imgs_dir)

    _shape = page_shape

    with open(book_page_tags_file, "w", encoding="utf-8") as fw:
        for i in range(init_num, init_num + obj_num):
            '''
            if page_shape is None and text_type == "h":
                _shape = (random.randint(480, 720), random.randint(640, 960))
            if page_shape is None and text_type == "v":
                _shape = (random.randint(640, 960), random.randint(480, 720))
            '''
            if page_shape is None:
                _shape = random.choice(BOOK_PAGE_SHAPE_LIST)

            PIL_page, text_bbox_list, split_pos_list = create_book_page_with_img(_shape, text_type=text_type)
            image_tags = {"text_bbox_list": text_bbox_list, "split_pos_list": split_pos_list}

            img_name = "book_page_%d.jpg" % i
            save_path = os.path.join(book_page_imgs_dir, img_name)
            PIL_page.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")

            if i % 50 == 0:
                print(" %d / %d Done" % (i, obj_num))
                sys.stdout.flush()


def create_book_page(shape=(960, 540), text=None, text_type="horizontal"):
    text_type = check_text_type(text_type)

    # 黑色背景书页
    np_page = np.zeros(shape=shape, dtype=np.uint8)
    page_height, page_width = shape

    # 随机确定是否画边框线及行线
    draw = None
    if random.random() < 0.7:
        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

    # 随机确定书页边框
    margin_w = round(random.uniform(0.01, 0.05) * page_width)
    margin_h = round(random.uniform(0.01, 0.05) * page_height)
    margin_line_thickness = random.randint(2, 6)
    line_thickness = round(margin_line_thickness / 2)
    if draw is not None:
        # 点的坐标格式为(x, y)，不是(y, x)
        draw.rectangle([(margin_w, margin_h), (page_width - 1 - margin_w, page_height - 1 - margin_h)],
                       fill=None, outline="white", width=margin_line_thickness)

    # 记录下文本行的bounding-box
    text_bbox_records_list = []
    head_tail_list = []

    if text_type == "h":  # 横向排列

        # 随机确定文本的行数
        rows_num = random.randint(6, 10)
        row_h = (page_height - 2 * margin_h) / rows_num

        # y-coordinate划分行
        ys = [margin_h + round(i * row_h) for i in range(rows_num)] + [page_height - 1 - margin_h]

        # 画行线，第一条线和最后一条线是边框线，不需要画
        if draw is not None:
            for y in ys[1:-1]:
                draw.line([(margin_w, y), (page_width - 1 - margin_w, y)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)

        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.02, 0.15), random.uniform(0.0, 0.2))  # (高方向, 宽方向)

        # 逐行生成汉字
        for i in range(len(ys) - 1):
            y1, y2 = ys[i] + 1, ys[i + 1] - 1
            x = margin_w + int(random.uniform(0.0, 1) * margin_line_thickness)
            row_length = page_width - x - margin_w
            _, text_bbox_list, _ = generate_mix_rows_chars(x, y1, y2, row_length, np_page, char_spacing, text=text)
            text_bbox_records_list.extend(text_bbox_list)
            if len(text_bbox_list) == 2:
                head_tail_list.extend([(text_bbox_list[0][1], text_bbox_list[0][3]),
                                       (text_bbox_list[1][1], text_bbox_list[1][3])])
            else:
                min_y1 = min([_y1 for _x1, _y1, _x2, _y2 in text_bbox_list])
                max_y2 = max([_y2 for _x1, _y1, _x2, _y2 in text_bbox_list])
                head_tail_list.append((min_y1, max_y2))

        # 获取行之间的划分位置
        split_pos = [margin_h, ]
        for i in range(len(head_tail_list) - 1):
            y_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
            split_pos.append(y_cent)
        split_pos.append(page_height - 1 - margin_h)

    else:  # 纵向排列

        # 随机决定文本的列数
        # cols_num = random.randint(6, 10)
        # cols_num = random.randint(18, 23)
        # cols_num = random.randint(14, 19)
        cols_num = random.randint(16, 20)
        col_w = (page_width - 2 * margin_w) / cols_num

        # x-coordinate划分列
        xs = [margin_w + round(i * col_w) for i in range(cols_num)] + [page_width - 1 - margin_w, ]

        # 画列线，第一条线和最后一条线是边缘线，不需要画
        if draw is not None:
            for x in xs[1:-1]:
                draw.line([(x, margin_h), (x, page_height - 1 - margin_h)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)

        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)

        # 逐列生成汉字，最右边为第一列
        for i in range(len(xs) - 1, 0, -1):
            x1, x2 = xs[i - 1] + 1, xs[i] - 1
            y = margin_h + int(random.uniform(0.0, 1) * margin_line_thickness)
            col_length = page_height - y - margin_h
            _, text_bbox_list, _ = generate_mix_cols_chars(x1, x2, y, col_length, np_page, char_spacing, text=text)
            text_bbox_records_list.extend(text_bbox_list)

            if len(text_bbox_list) == 2:
                head_tail_list.extend([(text_bbox_list[1][0], text_bbox_list[1][2]),
                                       (text_bbox_list[0][0], text_bbox_list[0][2])])
            else:
                min_x1 = min([_x1 for _x1, _y1, _x2, _y2 in text_bbox_list])
                max_x2 = max([_x2 for _x1, _y1, _x2, _y2 in text_bbox_list])
                head_tail_list.append((min_x1, max_x2))

        head_tail_list.reverse()  # 由于最右边为第一列，需要反转

        # 获取列之间的划分位置
        split_pos = [margin_w, ]
        for i in range(len(head_tail_list) - 1):
            x_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
            split_pos.append(x_cent)
        split_pos.append(page_width - 1 - margin_w)

    # 将黑底白字转换为白底黑字
    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    # print(text_bbox_list)
    # print(len(text_bbox_list))
    # PIL_page.show()

    return PIL_page, text_bbox_records_list, split_pos


def create_book_page_with_img(shape=(960, 540), text_type="horizontal"):
    text_type = check_text_type(text_type)

    # 黑色背景书页
    np_page = np.zeros(shape=shape, dtype=np.uint8)
    page_height, page_width = shape

    # 随机确定是否画边框线及行线
    draw = None
    if random.random() < 0.7:
        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

    # 随机确定书页边框
    margin_w = round(random.uniform(0.01, 0.05) * page_width)
    margin_h = round(random.uniform(0.01, 0.05) * page_height)
    margin_line_thickness = random.randint(2, 6)
    line_thickness = round(margin_line_thickness / 2)
    if draw is not None:
        # 点的坐标格式为(x, y)，不是(y, x)
        draw.rectangle([(margin_w, margin_h), (page_width - 1 - margin_w, page_height - 1 - margin_h)],
                       fill=None, outline="white", width=margin_line_thickness)

    # 记录下文本行的bounding-box
    text_bbox_records_list = []
    head_tail_list = []

    if text_type == "h":  # 横向排列

        # 随机确定文本的行数
        rows_num = random.randint(6, 10)
        row_h = (page_height - 2 * margin_h) / rows_num

        # y-coordinate划分行
        ys = [margin_h + round(i * row_h) for i in range(rows_num)] + [page_height - 1 - margin_h]

        # 画行线，第一条线和最后一条线是边框线，不需要画
        if draw is not None:
            for y in ys[1:-1]:
                draw.line([(margin_w, y), (page_width - 1 - margin_w, y)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)

        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.02, 0.15), random.uniform(0.0, 0.2))  # (高方向, 宽方向)

        # 逐行生成汉字
        for i in range(len(ys) - 1):
            y1, y2 = ys[i] + 1, ys[i + 1] - 1
            x = margin_w + int(random.uniform(1.0, 1.5) * margin_line_thickness)
            row_length = page_width - x - margin_w
            if random.random() < 0.3:
                row_length = int(random.uniform(0.2, 1) * row_length)
                if draw is not None:
                    draw.line([(x, y1), (x, y2)], fill="white", width=line_thickness)
            _, text_bbox_list, _ = generate_mix_rows_chars(x, y1, y2, row_length, np_page, char_spacing, use_img=True)
            text_bbox_records_list.extend(text_bbox_list)
            if len(text_bbox_list) == 2:
                head_tail_list.extend([(text_bbox_list[0][1], text_bbox_list[0][3]),
                                       (text_bbox_list[1][1], text_bbox_list[1][3])])
            else:
                min_y1 = min([_y1 for _x1, _y1, _x2, _y2 in text_bbox_list])
                max_y2 = max([_y2 for _x1, _y1, _x2, _y2 in text_bbox_list])
                head_tail_list.append((min_y1, max_y2))

        # 获取行之间的划分位置
        split_pos = [margin_h, ]
        for i in range(len(head_tail_list) - 1):
            y_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
            split_pos.append(y_cent)
        split_pos.append(page_height - 1 - margin_h)

    else:  # 纵向排列

        # 随机决定文本的列数
        # cols_num = random.randint(6, 10)
        # cols_num = random.randint(18, 23)
        # cols_num = random.randint(14, 19)
        cols_num = random.randint(16, 20)
        col_w = (page_width - 2 * margin_w) / cols_num

        # x-coordinate划分列
        xs = [margin_w + round(i * col_w) for i in range(cols_num)] + [page_width - 1 - margin_w, ]

        # 画列线，第一条线和最后一条线是边缘线，不需要画
        if draw is not None:
            for x in xs[1:-1]:
                draw.line([(x, margin_h), (x, page_height - 1 - margin_h)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)

        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)

        ys = [0 for i in range(len(xs))]
        col_lengths = [0 for i in range(len(xs))]

        for i in range(len(xs) - 1, 0, -1):
            x1, x2 = xs[i - 1] + 1, xs[i] - 1
            ys[i] = margin_h + int(random.uniform(0.0, 1) * margin_line_thickness)
            col_lengths[i] = page_height - ys[i] - margin_h
            if random.random() < 0.3:
                col_lengths[i] = int(random.uniform(0.2, 1) * col_lengths[i])
                yy = col_lengths[i] + margin_h + margin_line_thickness
                if draw is not None:
                    draw.line([(x1, yy), (x2, yy)], fill="white", width=line_thickness)
                    np_page = np.array(PIL_page, dtype=np.uint8)
                    # Image.fromarray(np_page).show()

        # 逐列生成汉字，最右边为第一列
        for i in range(len(xs) - 1, 0, -1):
            x1, x2 = xs[i - 1] + 1, xs[i] - 1
            y = ys[i]
            col_length = col_lengths[i]
            _, text_bbox_list, _ = generate_mix_cols_chars(x1, x2, y, col_length, np_page, char_spacing, use_img=True)
            text_bbox_records_list.extend(text_bbox_list)

            if len(text_bbox_list) == 2:
                head_tail_list.extend([(text_bbox_list[1][0], text_bbox_list[1][2]),
                                       (text_bbox_list[0][0], text_bbox_list[0][2])])
            else:
                min_x1 = min([_x1 for _x1, _y1, _x2, _y2 in text_bbox_list])
                max_x2 = max([_x2 for _x1, _y1, _x2, _y2 in text_bbox_list])
                head_tail_list.append((min_x1, max_x2))

        head_tail_list.reverse()  # 由于最右边为第一列，需要反转

        # 获取列之间的划分位置
        split_pos = [margin_w, ]
        for i in range(len(head_tail_list) - 1):
            x_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
            split_pos.append(x_cent)
        split_pos.append(page_width - 1 - margin_w)

    # 将黑底白字转换为白底黑字
    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    # print(text_bbox_list)
    # print(len(text_bbox_list))
    # PIL_page.show()

    return PIL_page, text_bbox_records_list, split_pos


if __name__ == '__main__':
    text = '春江潮水连海平，海上明月共潮生'
    text_queue = Queue()
    for char in text:
        text_queue.put(char)
    generate_book_page_imgs(obj_num=5, text=text_queue, text_type="vertical")
    # generate_book_page_imgs(obj_num=10, text=None, text_type="vertical")
    # generate_book_page_imgs_with_img(obj_num=10, text_type="vertical")
    # generate_book_page_imgs(obj_num=5, text_type="vertical")
