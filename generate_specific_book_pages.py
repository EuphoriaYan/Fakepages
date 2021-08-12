# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

from pprint import pprint
import json
import random
import cv2
import re
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from queue import Queue

from fontTools.ttLib import TTFont
from torchvision import transforms
from torch import nn
import copy

from util import IMPORTANT_CHARS, SMALL_IMPORTANT_CHARS, UNDERLINE_CHAR
from config import config_manager, FONT_FILE_DIR

from img_utils import rotate_PIL_image, find_min_bound_box, adjust_img_and_put_into_background, reverse_image_color, \
    edge_distortion, set_config
from noise_util import add_noise, white_erosion, triangle_contrast
from ocrodeg import *
from generate_font_sample import create_mix_ch_handle, create_imgs_ch_handle, create_ttf_ch_handle
from generate_seal import change_seal_color, change_seal_shape


def check_text_type(text_type):
    if text_type.lower() in ("h", "horizontal"):
        text_type = "h"
    elif text_type.lower() in ("v", "vertical"):
        text_type = "v"
    else:
        ValueError("Optional text_types: 'h', 'horizontal', 'v', 'vertical'.")
    return text_type


class generate_text_lines_with_text_handle:

    # def __init__(self, obj_num, shape=None, text_type="horizontal", text='野火烧不尽春风吹又生', char_size=64,
    #              augment=True, fonts_json='/disks/sdb/projs/AncientBooks/data/chars/font_missing.json',
    #              fonts_root=None, bad_font_file='charset/songhei_error_font.txt',
    #              experiment_dir='songhei_experiment/', type_fonts='type/宋黑类字符集.txt', embedding_num=250,
    #              resume=70000, charset='charset/charset_xl.txt', init_num=0, special_type='normal',
    #              segment_type='normal'):
    def __init__(self, config):
        if config.char_from == 'fontmagic':
            self.generate_char_handle = create_mix_ch_handle(
                fonts_json=config.fonts_json,
                fonts_root=config.fonts_root,
                bad_font_file=config.bad_font_file,
                experiment_dir=config.font_magic_experiment,
                type_fonts=config.type_fonts,
                embedding_num=config.embedding_num,
                resume=config.resume
            )
        elif config.char_from == 'ttf':
            self.generate_char_handle = create_ttf_ch_handle(
                ttf_path=config.ttf_path,
                default_ttf_path=config.default_ttf_path,
                char_size=config.char_size,
                canvas_size=config.canvas_size
            )
        elif config.char_from == 'imgs':  # TODO need fix
            self.generate_char_handle = create_imgs_ch_handle(
                config.char_imgs_path
            )
        # 生成印章只能用ttf
        if config.seal_page:
            self.generate_char_handle = create_ttf_ch_handle(
                ttf_path=config.seal_ttf_path,
                default_ttf_path=config.default_ttf_path,
                char_size=config.char_size,
                canvas_size=config.canvas_size
            )
        self.config = config

    def generate_book_page_with_text(self):
        config = self.config
        orient = config.orient  # 确定排列方向（横or竖）
        os.makedirs(config.store_imgs, exist_ok=True)

        with open(config.store_tags, "w", encoding="utf-8") as fw:
            for i in range(config.init_num, config.obj_num):
                if isinstance(config.shape, list):
                    shape = random.choice(config.shape)
                else:
                    shape = config.shape

                if config.seal_page:  # 1/8的概率生成半阴半阳的印章
                    if random.random() < 1/8:
                        config.multiple_plate = True
                        config.plate_type = 'split_vertical'
                    else:
                        config.multiple_plate = False

                if config.multiple_plate:
                    PIL_page, text_bbox_list, text_list, char_bbox_list, char_list = self.create_multiple_plate(
                        shape, orient, config.plate_type
                    )
                else:  # 不需要拼接，单页即可
                    self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变

                    PIL_page, text_bbox_list, text_list, char_bbox_list, char_list, _ = self.create_book_page_with_text(
                        shape, orient
                    )
                # if config.seal_in_page:  # 创建印章
                #     PIL_page_seal, text_bbox_list, char_bbox_list = self.add_seal(shape, text_bbox_list, char_bbox_list)

                if config.seal_page:  # 给印章调整形状
                    PIL_page, text_bbox_list, char_bbox_list = change_seal_shape(PIL_page, text_bbox_list, char_bbox_list, config.multiple_plate)

                if config.contrast:
                    for j in range(1, 20):
                        PIL_page = triangle_contrast(PIL_page)

                if config.edge_distortion:
                        PIL_page = edge_distortion(PIL_page)

                if config.augment:
                    new_text_bbox_list = []
                    new_char_bbox_list = []
                    if random.random() > 0.5:
                        w, h = PIL_page.size
                        # resize_ratio_range = (0.8, 1.3)
                        resize_ratio_range = 1.11
                        resize_ratio = random.uniform(1, resize_ratio_range)
                        # resize_ratio = np.exp(resize_ratio)
                        new_w = int(w / resize_ratio)
                        new_h = int(h * resize_ratio)
                        PIL_page = PIL_page.resize((new_w, new_h))
                        for text_bbox in text_bbox_list:
                            new_text_bbox = (
                                int(text_bbox[0] / resize_ratio),
                                int(text_bbox[1] * resize_ratio),
                                int(text_bbox[2] / resize_ratio),
                                int(text_bbox[3] * resize_ratio),
                            )
                            new_text_bbox_list.append(new_text_bbox)
                        for char_bbox in char_bbox_list:
                            new_char_bbox = (
                                int(char_bbox[0] / resize_ratio),
                                int(char_bbox[1] * resize_ratio),
                                int(char_bbox[2] / resize_ratio),
                                int(char_bbox[3] * resize_ratio),
                            )
                            new_char_bbox_list.append(new_char_bbox)
                        text_bbox_list = new_text_bbox_list
                        char_bbox_list = new_char_bbox_list
                    if config.seal_page:  # 印章的噪音多一些
                        PIL_page = white_erosion(PIL_page)
                        PIL_page = add_noise(PIL_page, 0.006, 0.03)
                    else:
                        PIL_page = add_noise(PIL_page)
                    PIL_page = ocrodeg_augment(PIL_page, seal=config.seal_page)

                if config.seal_page:  # 给印章上色
                    PIL_page = change_seal_color(PIL_page)

                # if config.seal_in_page: # 添加印章
                #     PIL_page = PIL_page.convert('RGB')
                #     PIL_page_seal = PIL_page_seal.convert('RGB')
                #     PIL_page_seal = PIL_page_seal.resize(PIL_page.size)
                #     PIL_page = Image.blend(PIL_page, PIL_page_seal, 0.4)
                if config.page_color == 'black':
                    PIL_page = reverse_image_color(PIL_img=PIL_page)
                if config.page_color == 'random' and random.random() < 0.5:
                    PIL_page = reverse_image_color(PIL_img=PIL_page)

                image_tags = {"text_bbox_list": text_bbox_list, "text_list": text_list,
                              "char_bbox_list": char_bbox_list, "char_list": char_list}

                img_name = "book_page_%d.jpg" % i
                save_path = os.path.join(config.store_imgs, img_name)
                PIL_page.save(save_path, format="jpeg")
                fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")

                if i % 50 == 0:
                    print(" %d / %d Done" % (i, self.config.obj_num))
                    sys.stdout.flush()
    # def change_seal_form(self, PIL_pang, text_bbox_list, char_bbox_list):

    def create_multiple_plate(self, shape, orient, type):
        page_height, page_width = shape
        np_page = np.zeros(shape=shape, dtype=np.uint8)
        left_side = 0
        right_side = 0
        top_side = 0
        bottom_side = 0

        if type == 'split_vertical':
            page_width_R = random.randint(int(page_width*0.3), int(page_width*0.7))
            if config.seal_page:  # 印章对半分
                page_width_R = int(page_width*0.5)
            page_width_L = page_width - page_width_R - 1

            x_L1 = 0
            x_L2 = x_L1 + page_width_L
            x_R1 = x_L2 + 1
            x_R2 = page_width

            shape_R = (page_height, page_width_R)
            shape_L = (page_height, page_width_L)

            self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
            PIL_page_R, text_bbox_list_R, text_list_R, char_bbox_list_R, char_list_R, col_w = self.create_book_page_with_text(
                shape_R, orient, margin_at_left=False, draw_frame=False)
            PIL_page_L, text_bbox_list_L, text_list_L, char_bbox_list_L, char_list_L, _ = self.create_book_page_with_text(
                shape_L, orient, margin_at_right=False, draw_frame=False, col_w=col_w)

            # 印章一半一半阴阳
            if config.seal_page:
                if random.random() < 0.5:
                    PIL_page_L = reverse_image_color(PIL_img=PIL_page_L)
                else:
                    PIL_page_R = reverse_image_color(PIL_img=PIL_page_R)

            np_page, text_bbox_list_R, char_bbox_list_R = \
                self.add_subpage_into_page(np_page, PIL_page_R, text_bbox_list_R, char_bbox_list_R,
                                           x_R1, x_R2, 0, page_height)
            np_page, text_bbox_list_L, char_bbox_list_L = \
                self.add_subpage_into_page(np_page, PIL_page_L, text_bbox_list_L, char_bbox_list_L,
                                           x_L1, x_L2, 0, page_height)

            np_page = reverse_image_color(np_img=np_page)
            PIL_page = Image.fromarray(np_page)

            # 合并tag
            text_bbox_list = text_bbox_list_R + text_bbox_list_L
            text_list = text_list_R + text_list_L
            char_bbox_list = char_bbox_list_R + char_bbox_list_L
            char_list = char_list_R + char_list_L

        if type == 'note_outside':
            page_width_note = random.randint(int(page_width/6), int(page_width/4))
            page_height_note = random.randint(int(page_height/6), int(page_height/4))
            if random.random() < 0.5:
                if random.random() < 0.5:  # note靠左
                    x_note1 = 0
                    x_note2 = x_note1 + page_width_note
                    left_side = page_width_note
                else:  # note靠右
                    x_note2 = page_width
                    x_note1 = x_note2 - page_width_note
                    right_side = page_width_note
                y_note1 = random.randint(0, page_height - page_height_note)
                y_note2 = y_note1 + page_height_note
            else:
                if random.random() < 0.5:  # note靠上
                    y_note1 = 0
                    y_note2 = y_note1 + page_height_note
                    top_side = page_height_note
                else:  # note靠下
                    y_note2 = page_height
                    y_note1 = y_note2 - page_height_note
                    bottom_side = page_height_note
                x_note1 = random.randint(0, page_width - page_width_note)
                x_note2 = x_note1 + page_width_note

            page_width_body = page_width - left_side - right_side - 1
            page_height_body = page_height - top_side - bottom_side - 1

            x_body1 = left_side + 1
            x_body2 = x_body1 + page_width_body
            y_body1 = top_side + 1
            y_body2 = y_body1 + page_height_body

            shape_body = (page_height_body, page_width_body)
            shape_note = (page_height_note, page_width_note)

            self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
            PIL_page_body, text_bbox_list_body, text_list_body, char_bbox_list_body, char_list_body, _ = self.create_book_page_with_text(
                shape_body, orient)
            self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
            PIL_page_note, text_bbox_list_note, text_list_note, char_bbox_list_note, char_list_note, _ = self.create_book_page_with_text(
                shape_note, orient, plat_type='note')

            np_page, text_bbox_list_body, char_bbox_list_body = self.add_subpage_into_page(
                np_page, PIL_page_body, text_bbox_list_body, char_bbox_list_body, x_body1, x_body2, y_body1, y_body2
            )
            np_page, text_bbox_list_note, char_bbox_list_note = self.add_subpage_into_page(
                np_page, PIL_page_note, text_bbox_list_note, char_bbox_list_note, x_note1, x_note2, y_note1, y_note2
            )

            np_page = reverse_image_color(np_img=np_page)
            PIL_page = Image.fromarray(np_page)

            if bottom_side != 0 or left_side != 0:
                text_bbox_list = text_bbox_list_body + text_bbox_list_note
                text_list =  text_list_body + text_list_note
                char_bbox_list = char_bbox_list_body + char_bbox_list_note
                char_list = char_list_body + char_list_note
            else:
                text_bbox_list = text_bbox_list_note + text_bbox_list_body
                text_list = text_list_note + text_list_body
                char_bbox_list = char_bbox_list_note + char_bbox_list_body
                char_list = char_list_note + char_list_body

        if type == 'note_inside':
            page_width_R = random.randint(int(page_width/6), int(page_width*0.4))  # note右侧板块的宽
            page_width_L = random.randint(int(page_width/6), int(page_width*0.4))  # note左侧板块的宽
            page_width_note = page_width - page_width_R - page_width_L - 2

            page_height_note = random.randint(int(page_height/4), int(page_height/2))
            page_height_below = page_height - page_height_note - 1  # note下面那个板块的高
            page_height_side = page_height - random.randint(0, int(page_height/8))  # note两侧板块的高

            # 生成右侧板块
            shape_R = (page_height_side, page_width_R)
            self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
            PIL_page_R, text_bbox_list_R, text_list_R, char_bbox_list_R, char_list_R, col_w = self.create_book_page_with_text(
                shape_R, orient, margin_at_top=False, margin_at_left=False, draw_frame=False)

            # 为两个“半列”和一个“半行”腾地方
            col_w = int(col_w)
            hide_part1 = int(col_w * random.uniform(0.5, 0.8))  # 左边
            hide_part2 = int(col_w * random.uniform(0.5, 0.8))  # 右边
            hide_part3 = int(col_w * random.uniform(0.5, 0.8))  # 下边
            page_width_L = page_width_L - 2 * col_w + hide_part1 + hide_part2
            page_width_below = page_width_note + 2 * col_w - hide_part1 - hide_part2
            page_height_below = page_height_below - col_w + hide_part3
            shape_note = (page_height_note, page_width_note)
            shape_below = (page_height_below, page_width_below)
            shape_L = (page_height_side, page_width_L)

            # 生成左侧和下侧板块
            PIL_page_below, text_bbox_list_below, text_list_below, char_bbox_list_below, char_list_below, _ = self.create_book_page_with_text(
                shape_below, orient, margin_at_left=False, margin_at_top=False, margin_at_right=False, draw_frame=False, col_w=col_w)
            PIL_page_L, text_bbox_list_L, text_list_L, char_bbox_list_L, char_list_L, _ = self.create_book_page_with_text(
                shape_L, orient, margin_at_top=False, margin_at_right=False, draw_frame=False, col_w=col_w)

            # 确定板块坐标
            x_L1 = 0
            x_L2 = page_width_L
            y_L1 = page_height - page_height_side
            y_L2 = page_height

            x_note1 = x_L2 + col_w - hide_part1 + 1
            x_note2 = x_note1 + page_width_note
            y_note1 = 0
            y_note2 = page_height_note

            x_below1 = x_L2 + 1
            x_below2 = x_below1 + page_width_below
            y_below1 = y_note2 + col_w - hide_part3 + 1
            y_below2 = y_below1 + page_height_below

            x_R1 = x_below2 + 1
            x_R2 = x_R1 + page_width_R
            y_R1 = y_L1
            y_R2 = y_L2

            # 插入两个“半列”，字体与左、右、下 相同
            x1 = x_L2 + 1
            x2 = x1 + col_w
            y = y_L1
            col_length = y_note2 - y_L1
            char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            # 左侧
            self.generate_one_col_chars_with_text(
                x1, x2, y, col_length, np_page, char_spacing
            )
            # 右侧
            x1 = x_note2 - hide_part2
            x2 = x1 + col_w
            self.generate_one_col_chars_with_text(
                x1, x2, y, col_length, np_page, char_spacing
            )
            # 下方
            x = x_below1
            y2 = y_below1
            y1 = y2 - col_w
            length_single = x_below2 - x_below1
            self.generate_one_row_chars_with_text(
                x, y1, y2, length_single, np_page, char_spacing
            )

            # 换个字体后生成便签板块
            self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
            PIL_page_note, text_bbox_list_note, text_list_note, char_bbox_list_note, char_list_note, _ = self.create_book_page_with_text(
                shape_note, orient, margin_at_left=False, margin_at_right=False, margin_at_bottom=False, plat_type='note'
            )

            np_page, text_bbox_list_note, char_bbox_list_note = self.add_subpage_into_page(
                np_page, PIL_page_note, text_bbox_list_note, char_bbox_list_note, x_note1, x_note2, y_note1, y_note2, cover=True
            )
            np_page, text_bbox_list_R, char_bbox_list_R = self.add_subpage_into_page(
                np_page, PIL_page_R, text_bbox_list_R, char_bbox_list_R, x_R1, x_R2, y_R1, y_R2
            )
            np_page, text_bbox_list_L, char_bbox_list_L = self.add_subpage_into_page(
                np_page, PIL_page_L, text_bbox_list_L, char_bbox_list_L, x_L1, x_L2, y_L1, y_L2
            )
            np_page, text_bbox_list_below, char_bbox_list_below = self.add_subpage_into_page(
                np_page, PIL_page_below, text_bbox_list_below, char_bbox_list_below, x_below1, x_below2, y_below1, y_below2
            )

            np_page = reverse_image_color(np_img=np_page)
            PIL_page = Image.fromarray(np_page)

            # 合并tag
            text_bbox_list = text_bbox_list_R + text_bbox_list_note + text_bbox_list_below + text_bbox_list_L
            text_list = text_list_R + text_list_note + text_list_below + text_list_L
            char_bbox_list = char_bbox_list_R + char_bbox_list_note + char_bbox_list_below + char_bbox_list_L
            char_list = char_list_R + char_list_note + char_list_below + char_list_L

        return PIL_page, text_bbox_list, text_list, char_bbox_list, char_list

    # def add_seal(self, shape, text_bbox_list, char_bbox_list):
    #     self.generate_char_handle = create_ttf_ch_handle(
    #         ttf_path=config.seal_ttf_path,
    #         default_ttf_path=config.default_ttf_path,
    #         char_size=config.char_size,
    #         canvas_size=config.canvas_size
    #     )
    #
    #     page_height, page_width = shape
    #     np_page = np.zeros(shape=shape, dtype=np.uint8)
    #
    #     seal_width = random.randint(int(page_width/8), int(page_height/4))
    #
    #     if random.random() < 0.3:  # 生成方形的印章
    #         seal_height = seal_width
    #     else:
    #         seal_height = seal_width * random.randint(1, 3)
    #     shape_seal = (seal_height, seal_width)
    #
    #     col_w = seal_width - 5
    #
    #     self.generate_char_handle.update()  # 更新生成单字的handle，切换当前字体/书法类型，一页一变
    #     PIL_page_seal, text_bbox_list_seal, text_list_seal, char_bbox_list_seal, char_list_seal, _ = self.create_book_page_with_text(
    #         shape_seal, 'vertical', margin_at_top=False, margin_at_bottom=False,
    #         margin_at_left=False, margin_at_right=False, draw_frame=False, col_w=col_w)
    #
    #     np_page_seal = np.array(PIL_page_seal, dtype=np.uint8)
    #     np_page_seal = reverse_image_color(np_img=np_page_seal)
    #     PIL_page_seal = Image.fromarray(np_page_seal)
    #
    #     x1 = random.randint(0, page_width - seal_width - 1)
    #     x2 = x1 + seal_width
    #     y1 = random.randint(0, page_height - seal_height - 1)
    #     y2 = y1 + seal_height
    #
    #     np_page, text_bbox_list_seal, char_bbox_list_seal = self.add_subpage_into_page(
    #         np_page, PIL_page_seal, text_bbox_list_seal, char_bbox_list_seal, x1, x2, y1, y2
    #     )
    #
    #     text_bbox_list += text_bbox_list_seal
    #     char_list_seal += char_bbox_list_seal
    #
    #     np_page = reverse_image_color(np_img=np_page)
    #
    #     black_img = np.zeros(shape)
    #     black_img = reverse_image_color(np_img=black_img)
    #     arr = np.dstack([black_img, np_page, np_page])
    #
    #     PIL_page = Image.fromarray(arr.astype('uint8')).convert('RGB')
    #
    #     self.__init__(config)
    #
    #     return PIL_page, text_bbox_list, char_bbox_list

    def add_subpage_into_page(self, np_page, PIL_subpage,
                              text_bbox_list, char_bbox_list, x1, x2, y1, y2, cover=False):
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

    def create_book_page_with_text(self, shape, orient, margin_at_top=True, margin_at_bottom=True,
                                   margin_at_left=True, margin_at_right=True, draw_frame=True, plat_type='', col_w=0):

        config = self.config

        # 黑色背景书页
        np_page = np.zeros(shape=shape, dtype=np.uint8)
        page_height, page_width = shape

        # 随机确定是否画边框线及行线
        if random.random() < 0.7 and config.draw_line:
            draw_line = True
        else:
            draw_line = False
        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

        # 随机确定书页边框
        margin_w = round(random.uniform(0.01, 0.05) * page_width)
        margin_h = round(random.uniform(0.01, 0.05) * page_height)
        margin_left = margin_w
        margin_right = margin_w
        margin_top = margin_h
        margin_bottom = margin_h
        if not margin_at_top:
            margin_top = 0
        if not margin_at_bottom:
            margin_bottom = 0
        if not margin_at_left:
            margin_left = 0
        if not margin_at_right:
            margin_right = 0
        margin_line_thickness = random.randint(2, 6)
        line_thickness = round(margin_line_thickness / 2)
        if draw_line and draw_frame:
            # 点的坐标格式为(x, y)，不是(y, x)
            draw.rectangle([(margin_left, margin_top), (page_width - 1 - margin_right, page_height - 1 - margin_bottom)],
                           fill=None, outline="white", width=margin_line_thickness)

        # 记录下文本行的bounding-box
        text_bbox_records_list = []
        text_records_list = []

        # 记录下单字的boundding-box
        char_bbox_records_list = []
        char_records_list = []

        if orient == 'horizontal':  # 横向排列

            # 随机确定文本的行数
            rows_num = random.randint(int(page_width / page_width * config.line_num[0]),
                                      int(page_width / page_width * config.line_num[1]))
            row_h = (page_height - 2 * margin_h) / rows_num

            # y-coordinate划分行
            ys = [margin_h + round(i * row_h) for i in range(rows_num)] + [page_height - 1 - margin_h]

            # 画行线，第一条线和最后一条线是边框线，不需要画
            if draw_line:
                for y in ys[1:-1]:
                    draw.line([(margin_w, y), (page_width - 1 - margin_w, y)], fill="white", width=line_thickness)
                np_page = np.array(PIL_page, dtype=np.uint8)

            # 随机决定字符间距占列距的比例
            if config.segment_type == 'normal':
                char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif config.segment_type == 'crowded':
                char_spacing = (random.uniform(-0.1, 0.1), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif config.segment_type == 'spacious':
                char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif config.segment_type == 'mixed':
                rand_num = random.random()
                if rand_num > 0.5:  # 50% crowded
                    char_spacing = (random.uniform(-0.1, 0.1), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                elif rand_num < 0.2:  # 20% spacious
                    char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                else:  # 30% normal
                    char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            else:
                raise ValueError

            # 逐行生成汉字
            for i in range(len(ys) - 1):
                y1, y2 = ys[i] + 1, ys[i + 1] - 1
                x = margin_w + int(random.uniform(0.0, 1) * margin_line_thickness)
                if config.full_line:
                    row_length = page_height - y - margin_h
                else:
                    line_length = random.uniform(config.line_length, 1)
                    row_length = int(line_length * page_height) - y - margin_h
                _, text_bbox_list, text_list, char_bbox_list, char_list = self.generate_mix_rows_chars_with_text(
                    x, y1, y2, row_length, np_page, char_spacing
                )
                text_bbox_records_list.extend(text_bbox_list)
                text_records_list.extend(text_list)
                char_bbox_records_list.extend(char_bbox_list)
                char_records_list.extend(char_list)

        elif orient == 'vertical':  # 纵向排列

            # 随机决定文本的列数
            cols_num = random.randint(int(page_width / page_height * config.line_num[0]),
                                      int(page_width / page_height * config.line_num[1]))
            # 便条的列数少一些
            if plat_type == 'note':
                cols_num = random.randint(int(page_width / page_height * config.line_num[0] / 2),
                                          int(page_width / page_height * config.line_num[1] / 2))
            if cols_num == 0:
                cols_num = 1
            # 若无设定列宽，则按照随机列数决定
            if col_w == 0:
                col_w = (page_width - margin_left - margin_right) / cols_num
            else:  # 若有设定列宽，则列数重新取值，列宽取合适的近似值
                cols_num = int((page_width - margin_left - margin_right) / col_w)
                if cols_num == 0:
                    cols_num = 1
                col_w = (page_width - margin_left - margin_right) / cols_num

            # x-coordinate划分列
            xs = [margin_left + round(i * col_w) for i in range(cols_num)] + [page_width - 1 - margin_right]

            # 画列线，第一条线和最后一条线是边缘线，不需要画
            if draw_line:
                for x in xs[1:-1]:
                    draw.line([(x, margin_top), (x, page_height - 1 - margin_bottom)], fill="white", width=line_thickness)
                np_page = np.array(PIL_page, dtype=np.uint8)

            # 随机决定字符间距占列距的比例
            if config.segment_type == 'normal':
                char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif config.segment_type == 'crowded':
                char_spacing = (random.uniform(-0.1, 0), 0)  # (高方向, 宽方向)
            elif config.segment_type == 'spacious':
                char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            elif config.segment_type == 'mixed':
                rand_num = random.random()
                if rand_num > 0.5:  # 50% crowded
                    char_spacing = (random.uniform(-0.1, 0.1), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                elif rand_num < 0.2:  # 20% spacious
                    char_spacing = (random.uniform(0.3, 0.6), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
                else:  # 30% normal
                    char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.02, 0.15))  # (高方向, 宽方向)
            else:
                raise ValueError

            if config.text_save_punctuation:
                char_spacing = (char_spacing[0], random.uniform(0.2, 0.25))

            # 汉字上下分区
            region_num = random.randint(config.region_num[0], config.region_num[1])
            rn = 0
            surplus_height = page_height - margin_top - margin_bottom
            while rn < region_num:

                y_start = page_height - margin_top - margin_bottom - surplus_height

                # 随机确定分区高度
                region_height = int(surplus_height / (region_num - rn))
                float_height = int(region_height / 2)
                region_height = random.randint(region_height - float_height, region_height + float_height)
                if region_height > surplus_height or rn == region_num - 1:
                    region_height = surplus_height

                surplus_height = surplus_height - region_height - config.region_thickness
                rn += 1

                # 画分区的线
                if config.region_draw_line != 'no' and rn != 1:
                    line_thickness = random.randint(2, 4)
                    p = Image.fromarray(np_page)
                    draw = ImageDraw.Draw(p)
                    draw.line([(margin_left, y_start + margin_h - 1),
                               (page_width - 1 - margin_right, y_start + margin_h - 1)],
                              fill="white",
                              width=line_thickness)
                    if config.region_draw_line == 'double':
                        double_line_y = y_start - config.region_thickness + margin_h
                        draw.line([(margin_left, double_line_y), (page_width - 1 - margin_right, double_line_y)],
                                  fill="white",
                                  width=line_thickness)
                    if config.region_draw_line == 'mix' and random.random() < 0.5:
                        double_line_y = y_start - config.region_thickness + margin_h
                        draw.line([(margin_left, double_line_y), (page_width - 1 - margin_right, double_line_y)],
                                  fill="white",
                                  width=line_thickness)
                    np_page = np.array(p, dtype=np.uint8)

                # 逐列生成汉字，最右边为第一列
                for i in range(len(xs) - 1, 0, -1):
                    x1, x2 = xs[i - 1] + 1, xs[i] - 1
                    y = margin_top + y_start + int(random.uniform(0.0, 1) * margin_line_thickness)
                    col_length = region_height

                    symbol_position = True
                    if config.symbol_position_specified:  # 用于判断左右端是否随机插入blank
                        min_edge = 0
                        max_edge = cols_num
                        if config.symbol_at_left:
                            min_edge += 3
                        if config.symbol_at_right:
                            max_edge -= 3
                        if i <=max_edge and i >= min_edge:
                            symbol_position = False
                    # 判断目前该行在整页的哪个部分
                    if i <= 3:
                        line_part = 'left'
                    elif i <= cols_num - 3:
                        line_part = 'center'
                    else:
                        line_part = 'right'

                    blank_length = int(config.blank_length * page_height - y - margin_h)
                    if config.blank_at_top:
                        if line_part in config.blank_at_top_lines:
                            y += blank_length
                            col_length -= blank_length
                        # 自定义留白
                        elif 'defined' in config.blank_at_top_lines:
                            if i >= cols_num * config.blank_at_top_defined[0] and i <= cols_num * config.blank_at_top_defined[1]:
                                y += blank_length
                                col_length -= blank_length
                    if config.blank_at_bottom:
                        if line_part in config.blank_at_bottom_lines:
                            col_length -= blank_length
                        # 自定义留白
                        elif 'defined' in config.blank_at_bottom_lines:
                            if i >= cols_num * config.blank_at_bottom_defined[0] and i <= cols_num * config.blank_at_bottom_defined[1]:
                                col_length -= blank_length

                    # 不填满整行
                    if not config.full_line:
                        line_length = random.uniform(config.line_length, 1)
                        col_length = int(line_length * col_length)

                    # 拉伸字体来充满整行（印章用）
                    if config.full_line_reshape:
                        char_num = random.randint(config.char_num_in_line[0], config.char_num_in_line[1])
                        config.char_reshape = True
                        config.char_reshape_line = 'both'
                        char_height = int(col_length / char_num)
                        config.char_single_line_reshape_stretch = char_height/(x2-x1+1)
                        config.char_single_line_reshape_stretch = char_height/(x2-x1+1)

                    if config.line_type == 'mixed':
                        _, text_bbox_list, text_list, char_bbox_list, char_list = self.generate_mix_cols_chars_with_text(
                            x1, x2, y, col_length, np_page, char_spacing, symbol_position=symbol_position
                        )
                    elif config.line_type == 'single':
                        _, text_bbox, text, char_bbox = self.generate_one_col_chars_with_text(
                            x1, x2, y, col_length, np_page, char_spacing
                        )
                        text_bbox_list = [text_bbox]
                        text_list = [text]

                        char_bbox_list = []
                        char_list = []
                        for i in range(0, len(text)):
                            if text[i] not in SMALL_IMPORTANT_CHARS:
                                char_bbox_list.append(char_bbox[i])
                                char_list.extend(text[i])

                        # char_bbox_list = char_bbox
                        # char_list = text
                    else:
                        raise ValueError



                    text_bbox_records_list.extend(text_bbox_list)
                    text_records_list.extend(text_list)
                    char_bbox_records_list.extend(char_bbox_list)
                    char_records_list.extend(char_list)

        # 向页面里添加图片
        if config.chart_in_page:
            for chart_position in config.chart_position_x_y:
                chart_name = config.chart_use[random.randint(0, 8)]
                chart_file = self.chart_dict(chart_name)
                PIL_chart_img = Image.open(chart_file).convert('L')

                chart_width = int(config.chart_size_to_page_w * (page_width - margin_w * 2))
                chart_height = int(config.chart_size_to_page_h * (page_height - margin_h * 2))
                PIL_chart_img = PIL_chart_img.resize((chart_width, chart_height))
                # 转为numpy格式
                np_chart_img = np.array(PIL_chart_img, dtype=np.uint8)
                # 黑白反色
                np_chart_img = reverse_image_color(np_img=np_chart_img)
                PIL_chart_img = Image.fromarray(np_chart_img)

                # 随机旋转
                if random.random() < 0.35:
                    PIL_chart_img = rotate_PIL_image(
                        PIL_chart_img,
                        rotate_angle=random.randint(-config.max_rotate_angle, config.max_rotate_angle)
                    )

                # 转为numpy格式
                np_chart_img = np.array(PIL_chart_img, dtype=np.uint8)

                chart_x, chart_y = chart_position
                chart_x1 = int((page_width - margin_w * 2) * chart_x) + margin_w
                chart_y1 = int((page_height - margin_h * 2) * chart_y) + margin_h
                chart_x2 = chart_x1 + chart_width
                chart_y2 = chart_y1 + chart_height

                # 将图片插入页面
                np_page[chart_y1:chart_y2, chart_x1:chart_x2] = np_chart_img

        # 将黑底白字转换为白底黑字
        np_page = reverse_image_color(np_img=np_page)
        PIL_page = Image.fromarray(np_page)
        return PIL_page, text_bbox_records_list, text_records_list, char_bbox_records_list, char_records_list, col_w

    def generate_mix_rows_chars_with_text(self, x, y1, y2, row_length, np_background, char_spacing):
        row_height = y2 - y1 + 1
        x_start = x

        text_bbox_list = []
        text_list = []
        char_bbox_list = []
        char_list = []
        flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
        remaining_len = row_length
        while remaining_len >= row_height:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(row_height, remaining_len)
            if config.limit_length_single != 0:
                length_single = random.randint(row_height, row_height * config.limit_length_single)
            else:
                length_single = length
            if config.limit_length_double != 0:
                length_double = random.randint(row_height, row_height * config.limit_length_double)
            else:
                length_double = length
            flag += 1
            if flag % 2 == 1 or config.line_type == 'single':
                x, text_bbox, text, char_bbox = self.generate_one_row_chars_with_text(
                    x, y1, y2, length_single, np_background, char_spacing
                )
                text_bbox_list.append(text_bbox)
                text_list.append(text)
                char_bbox_list.extend(char_bbox)
                char_list.extend(text)
            else:
                if 'split' in config.special_type:
                    x += length_double
                else:
                    x, text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2 = self.generate_two_rows_chars_with_text(
                        x, y1, y2, length_double, np_background, char_spacing
                    )
                    text_bbox_list.append(text1_bbox)
                    text_list.append(text1)
                    text_bbox_list.append(text2_bbox)
                    text_list.append(text2)
                    char_bbox_list.extend(char_bbox1)
                    char_list.extend(text1)
                    char_bbox_list.extend(char_bbox2)
                    char_list.extend(text2)
            remaining_len = row_length - (x - x_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        return x, text_bbox_list, text_list, char_bbox_list, char_list

    def generate_mix_cols_chars_with_text(self, x1, x2, y, col_length, np_background, char_spacing, symbol_position=True):
        col_width = x2 - x1 + 1
        y_start = y

        text_bbox_list = []
        text_list = []
        char_bbox_list = []
        char_list = []
        flag = 0 if random.random() < config.start_at_single else 1  # 以单行字串还是双行字串开始
        remaining_len = col_length

        if config.seal_page:
            last_char_judge = 0.8
        else:
            last_char_judge = 1

        while remaining_len >= last_char_judge * col_width:
            flag += 1
            if flag % 2 == 1:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                max_length = col_width * config.limit_max_length_single
                min_length = col_width * config.limit_min_length_single
                length = min(remaining_len, random.randint(min_length, max_length))

                # 该行结束，换行
                if config.end_at_single and flag == 1:
                    col_length = 0
                y, text_bbox, text, char_bbox = self.generate_one_col_chars_with_text(
                    x1, x2, y, length, np_background, char_spacing, symbol_position=symbol_position
                )
                text_bbox_list.append(text_bbox)
                text_list.append(text)

                for i in range(0, len(text)):
                    if text[i] not in SMALL_IMPORTANT_CHARS:
                        char_bbox_list.extend(char_bbox[i])
                        char_list.extend(text[i])

                # char_bbox_list.extend(char_bbox)
                # char_list.extend(text)
            else:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                max_length = col_width * config.limit_max_length_double
                min_length = col_width * config.limit_min_length_double
                length = min(remaining_len, random.randint(min_length, max_length))

                # 该行结束，换行
                if config.end_at_double:
                    col_length = 0

                if 'split' in self.config.special_type:
                    y += length
                else:
                    y, text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2 = self.generate_two_cols_chars_with_text(
                        x1, x2, y, length, np_background, char_spacing, symbol_position=symbol_position
                    )
                    text_bbox_list.append(text1_bbox)
                    text_list.append(text1)
                    text_bbox_list.append(text2_bbox)
                    text_list.append(text2)
                    char_bbox_list.extend(char_bbox1)
                    char_list.extend(text1)
                    char_bbox_list.extend(char_bbox2)
                    char_list.extend(text2)
            remaining_len = col_length - (y - y_start)

        # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...

        return y, text_bbox_list, text_list, char_bbox_list, char_list

    def generate_one_row_chars_with_text(self, x, y1, y2, length, np_background, char_spacing, line_type='single'):
        """
        :return: x, text_bbox, text
        """
        # 记录下生成的汉字及其bounding-box
        char_and_box_list = []

        row_end = x + length - 1
        row_height = y2 - y1 + 1
        while length >= row_height:
            if length < 1.5 * row_height:
                last_char = True
            else:
                last_char = False
            chinese_char, bounding_box, x_tail = self.generate_char_img_into_unclosed_box_with_text(
                np_background, x1=x, y1=y1, x2=None, y2=y2, char_spacing=char_spacing, last_char=last_char, line_type=line_type
            )

            char_and_box_list.append((chinese_char, bounding_box))
            added_length = x_tail - x
            length -= added_length
            x = x_tail

        # 获取文本行的bounding-box
        head_x1, head_y1, _, _ = char_and_box_list[0][1]
        _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
        text_bbox = (head_x1, head_y1, tail_x2, tail_y2)
        text = [char_and_box[0] for char_and_box in char_and_box_list]
        char_bbox = [char_and_box[1] for char_and_box in char_and_box_list]

        return x, text_bbox, text, char_bbox

    def generate_two_rows_chars_with_text(self, x, y1, y2, length, np_background, char_spacing):
        row_height = y2 - y1 + 1
        mid_y = y1 + round(row_height / 2)

        # x, text_bbox, text
        x_1, text1_bbox, text1, char_bbox1 = self.generate_one_row_chars_with_text(
            x, y1, mid_y, length, np_background, char_spacing, line_type='double')
        x_2, text2_bbox, text2, char_bbox2 = self.generate_one_row_chars_with_text(
            x, mid_y + 1, y2, length, np_background, char_spacing, line_type='double')

        return max(x_1, x_2), text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2

    def generate_one_col_chars_with_text(self, x1, x2, y, length, np_background, char_spacing, line_type='single',
                                         symbol_position=True):
        # 记录下生成的汉字及其bounding-box
        char_and_box_list = []

        col_end = y + length - 1
        col_width = x2 - x1 + 1
        first_char = True

        if config.seal_page:
            last_char_judge = 0.8
        else:
            last_char_judge = 1

        while length >= last_char_judge * col_width:
            last_char_seal = False
            # 就算字超过字框了，也不能让它超到页面外面去
            strech_to_full_line = 0
            if not first_char:
                if y + (char_spacing[0] + 1) * col_width >= np_background.shape[0]:
                    if config.full_line_reshape:
                        last_char_seal = True
                    else:
                        break
            if length < 1.5 * col_width:
                last_char = True
            else:
                last_char = False
            if last_char_seal:
                last_char = True
            if config.full_line_reshape and last_char:  # 印章专用，最后一个字拉长，撑满整行
                strech_to_full_line = length
            chinese_char, bounding_box, y_tail = self.generate_char_img_into_unclosed_box_with_text(
                np_background, x1=x1, y1=y, x2=x2, y2=None, char_spacing=char_spacing, first_char=first_char,
                last_char=last_char, line_type=line_type, symbol_position=symbol_position, strech_to_full_line=strech_to_full_line
            )
            if chinese_char is None:
                break
            char_and_box_list.append((chinese_char, bounding_box))
            added_length = y_tail - y
            length -= added_length
            y = y_tail
            first_char = False

        # 获取文本行的bounding-box
        head_x1, head_y1, _, _ = char_and_box_list[0][1]
        _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
        text_bbox = [head_x1, head_y1, tail_x2, tail_y2]
        text = [char_and_box[0] for char_and_box in char_and_box_list]
        char_bbox = [char_and_box[1] for char_and_box in char_and_box_list]

        return y, text_bbox, text, char_bbox

    def generate_two_cols_chars_with_text(self, x1, x2, y, length, np_background, char_spacing, symbol_position=True):
        col_width = x2 - x1 + 1
        mid_x = x1 + round(col_width / random.uniform(1.7, 2.3))

        y_1, text1_bbox, text1, char_bbox1 = self.generate_one_col_chars_with_text(
            mid_x + 1, x2, y, length, np_background, char_spacing, line_type='double', symbol_position=symbol_position)
        y_2, text2_bbox, text2, char_bbox2 = self.generate_one_col_chars_with_text(
            x1, mid_x, y, length, np_background, char_spacing, line_type='double', symbol_position=symbol_position)

        return max(y_1, y_2), text1_bbox, text2_bbox, text1, text2, char_bbox1, char_bbox2

    def generate_char_img_into_unclosed_box_with_text(self, np_background, x1, y1, x2=None, y2=None,
                                                      char_spacing=(0.05, 0.05), first_char=False, last_char=False,
                                                      line_type='', symbol_position=True, strech_to_full_line=0):
        config = self.config
        if x2 is None and y2 is None:
            raise ValueError("There is one and only one None in (x2, y2).")
        if x2 is not None and y2 is not None:
            raise ValueError("There is one and only one None in (x2, y2).")

        chinese_char = ' '
        PIL_char_img = None
        while PIL_char_img is None:
            if config.symbol_next_char and symbol_position:
                for symbol_next, symbol_next_prob in zip(config.symbol_next_use, config.symbol_next_prob):
                    if random.random() > symbol_next_prob:
                        continue
                    symbol_next_file = self.symbol_next_dict(symbol_next)
                    PIL_char_img = Image.open(symbol_next_file).convert('L')
                    break
            if PIL_char_img is None:
                if last_char and 'num_end' in config.special_type:
                    chinese_char = random.choice(['一', '二', '三'])
                else:
                    if config.text.empty():
                        config.init_text()
                    # 生成白底黑字的字，包含文字
                    if not config.text.empty():
                        chinese_char = config.text.get()
                        # print(config.charset)
                    while chinese_char not in config.charset:
                        if config.text.empty():
                            config.init_text()
                        chinese_char = config.text.get()
                        # print('hao')

                    # 第一个字不能是以下标点符号
                    while first_char and chinese_char in SMALL_IMPORTANT_CHARS:
                        if config.text.empty():
                            config.init_text()
                        chinese_char = config.text.get()
                PIL_char_img, flag = self.generate_char_handle.get_character(chinese_char)

        PIL_char_img = PIL_char_img.resize((config.char_size, config.char_size))

        # 随机决定是否对汉字图片进行旋转，以及旋转的角度
        if random.random() < 0.35:
            PIL_char_img = rotate_PIL_image(
                PIL_char_img,
                rotate_angle=random.randint(-config.max_rotate_angle, config.max_rotate_angle)
            )

        # 如果是纵向排版，这些标点符号旋转90度
        if config.orient == 'vertical' and chinese_char in set('《》（）〔〕［］【】〈〉<>'):
            PIL_char_img = rotate_PIL_image(PIL_char_img, rotate_angle=-90)

        # 转为numpy格式
        np_char_img = np.array(PIL_char_img, dtype=np.uint8)

        # if chinese_char in IMPORTANT_CHARS or chinese_char == ' ':
        #     pass
        if chinese_char == ' ':
            pass
        else:
            # 查找字体的最小包含矩形
            left, right, top, low = find_min_bound_box(np_char_img)
            np_char_img = np_char_img[top:low + 1, left:right + 1]

        char_img_height, char_img_width = np_char_img.shape[:2]

        if config.use_bigger_canvas:
            if line_type == 'double' and config.use_bigger_canvas_double is True:
                np_char_img = bigger_canvas(np_char_img)
            elif line_type == 'single' and config.use_bigger_canvas_single is True:
                np_char_img = bigger_canvas(np_char_img)

        # 添加与字图重叠的符号
        if config.symbol_on_char:
            for symbol, symbol_prob in zip(config.symbol_use, config.symbol_prob):
                if random.random() > symbol_prob:
                    continue
                symbol_file = config.symbol_dict[symbol]
                symbol_img = Image.open(symbol_file).convert('L')
                symbol_height, symbol_width = np_char_img.shape[:2]
                symbol_img = symbol_img.resize((symbol_width, symbol_height))
                symbol_arr = np.array(symbol_img)
                if symbol == 'reverse':
                    if config.use_bigger_canvas:
                        symbol_arr = bigger_canvas(symbol_arr)
                    np_char_img = bigger_canvas(np_char_img, shrink=0.5)
                    symbol_arr = reverse_image_color(np_img=symbol_arr)
                    np_char_img |= symbol_arr
                    np_char_img = reverse_image_color(np_img=np_char_img)
                else:
                    np_char_img |= symbol_arr

                if symbol in ["one_circle_white","one_circle_black", "two_circles", "underline", "wave"]:
                    break

        if chinese_char in SMALL_IMPORTANT_CHARS and y2 is None:  # 纵向排列时的标点符号
            col_w = x2 - x1 +1
            char_spacing_h = round(col_w * char_spacing[0])
            char_spacing_w = round(col_w * char_spacing[1])

            # 标点符号放在上一个字的右侧
            box_x1 = x2 - char_spacing_w
            box_x2 = x2

            box_w = box_x2 - box_x1 + 1
            box_h = min(round(char_img_height * box_w / char_img_width), round(0.8 * (col_w - char_spacing_w)))
            if box_h < round(0.5 * col_w):
                box_y1 = y1 - random.randint(box_h, round(0.5 * col_w)) + char_spacing_h
            else:
                box_y1 = y1 - random.randint(box_h, round(0.8 * col_w)) + char_spacing_h
            box_y2 = box_y1 + box_h - 1

            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))

        elif x2 is None:  # 文本横向排列
            row_h = y2 - y1 + 1
            char_spacing_h = round(row_h * char_spacing[0])
            char_spacing_w = round(row_h * char_spacing[1])
            if first_char:
                box_x1 = x1
            else:
                box_x1 = x1 + char_spacing_w
            box_y1 = y1 + char_spacing_h
            box_y2 = y2 - char_spacing_h
            box_h = box_y2 - box_y1 + 1

            # 是否需要将字拉伸为长方形
            stretch = 1.0
            if config.char_reshape:
                if config.char_reshape_line in [line_type, 'both']:
                    if line_type == 'single':
                        stretch = config.char_single_line_reshape_stretch
                    else:  # line_type == 'double'
                        stretch = config.char_double_line_reshape_stretch

            if char_img_height * 1.4 < char_img_width:
                # 对于“一”这种高度很小、宽度很大的字，应该生成正方形的字图片
                box_w = int(box_h * stretch)
                np_char_img = adjust_img_and_put_into_background(np_char_img, background_size_w=box_w, background_size_h=box_h)
            else:
                # 对于宽高相差不大的字，高度撑满，宽度随意
                box_w = int(round(char_img_width * box_h / char_img_height) * stretch)
                np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))

            box_x2 = box_x1 + box_w - 1

        else:  # y2 is None, 文本纵向排列
            col_w = x2 - x1 + 1
            char_spacing_h = round(col_w * char_spacing[0])
            char_spacing_w = round(col_w * char_spacing[1])
            box_x1 = x1 + char_spacing_w
            box_x2 = x2 - char_spacing_w
            if first_char:
                box_y1 = y1
            else:
                box_y1 = y1 + char_spacing_h
            box_w = box_x2 - box_x1 + 1

            # 是否需要将字拉伸为长方形
            stretch = 1.0
            if config.char_reshape:
                if config.char_reshape_line in [line_type, 'both']:
                    if line_type == 'single':
                        stretch = config.char_single_line_reshape_stretch
                    else:  # reshape == 'double'
                        stretch = config.char_double_line_reshape_stretch

            if char_img_width * 1.4 < char_img_height:
                # 对于“卜”这种高度很大、宽度很小的字，应该生成正方形的字图片
                box_h = int(box_w * stretch)
                if strech_to_full_line != 0:
                    box_h = strech_to_full_line
                if config.seal_page:  # 印章要撑满
                    np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
                else:
                    np_char_img = adjust_img_and_put_into_background(np_char_img, background_size_w=box_w, background_size_h=box_h)
            else:
                # 对于宽高相差不大的字，宽度撑满，高度随意
                box_h = int(round(char_img_height * box_w / char_img_width) * stretch)
                if strech_to_full_line != 0:
                    box_h = strech_to_full_line
                    if char_img_height * 1.4 < char_img_width:
                        np_char_img = adjust_img_and_put_into_background(np_char_img, background_size_w=box_w, background_size_h=box_h)
                np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))

            box_y2 = box_y1 + box_h - 1

        # 将生成的汉字图片放入背景图片
        try:
            # use 'or' function to make crowded imgs.
            np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] |= np_char_img
        except ValueError as e:
            print('Exception:', e)
            print("The size of char_img is larger than the length of (y1, x1) to edge. Now, resize char_img ...")
            if x2 is None:
                box_x2 = np_background.shape[1] - 1
                box_w = box_x2 - box_x1 + 1
            else:
                box_y2 = np_background.shape[0] - 1
                box_h = box_y2 - box_y1 + 1
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
            try:
                np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] |= np_char_img
            except ValueError as e:
                print('Exception:', e)
                print('Can\'t resize, ignore this char')
                return None, None, None

        # 随机选定汉字图片的bounding-box
        bbox_x1 = random.randint(x1, box_x1)
        if box_y1 > y1:
            bbox_y1 = random.randint(y1, box_y1)
        else:
            bbox_y1 = random.randint(box_y1, y1)
        bbox_x2 = min(random.randint(box_x2, box_x2 + char_spacing_w), np_background.shape[1] - 1)
        if char_spacing_h >= 0:
            bbox_y2 = min(random.randint(box_y2, box_y2 + char_spacing_h), np_background.shape[0] - 1)
        else:
            bbox_y2 = min(random.randint(box_y2 + char_spacing_h, box_y2), np_background.shape[0] - 1)
        bounding_box = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]

        # 包围标点符号的的最小box作为bounding-box
        if chinese_char in SMALL_IMPORTANT_CHARS:
            bounding_box = (box_x1, box_y1, box_x2, box_y2)

        char_box_tail = box_x2 + 1 if x2 is None else box_y2 + 1

        # 在纵向排版情况下，以下标点符号将会紧挨放在字的右侧，不占用位置
        if config.orient == 'vertical' and chinese_char in SMALL_IMPORTANT_CHARS:
            char_box_tail = y1

        return chinese_char, bounding_box, char_box_tail

    def generate_underline_right_side(self, PIL_page, text, char_bbox):
        part_num = random.randint(1, min(6, len(text)))
        part_edges = []

        # 划分边界
        for i in range(1, part_num):
            edge = random.randint(0, len(text))
            while edge in part_edges:
                edge = random.randint(0, len(text))

        # 按大小给边界排序
        part_edges.sort()




    def symbol_next_dict(self,symbol_name):
        config = self.config

        if symbol_name in config.symbol_number_next_char:
            name = symbol_name + str(random.randint(1, 10)) + '.PNG'
        else:
            name = symbol_name + '.png'
        return os.path.join(config.symbol_path, name)

    def chart_dict(self, chart_name):
        config = self.config

        num = random.randint(1, config.chart_max_num[chart_name])
        name = chart_name + '- (' + str(num) + ').jpg'
        return os.path.join(config.chart_path, name)

    def gen_font_labels_all(name_font, pth_font):
        font = TTFont(pth_font)
        unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
        out = ''
        for k,v in tqdm(unicode_map.items()):
            # 过滤
            # uni_lst = ['uni','u1','u2']
            # if not any(uni in v for uni in uni_lst) or len(v) < 6 : continue
            # out+=('{} ----> {}\n'.format(k, chr(int(v[3:],16))))
            # 不过滤
            out += ('{}\t{}\t{}\n'.format(k,v,chr(int(k))))
        pth_font_labels = os.path.join(FONT_FILE_DIR, name_font+'.labels.txt')
        with open(pth_font_labels, 'w+', encoding='utf-8') as f:
            f.write(out)

# 对字体图像做等比例缩放
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = args.config
    return config

def bigger_canvas(np_char_img, shrink = 1):
    half = int(0.5 * config.char_size)

    top = int(config.char_size * config.use_bigger_canvas_scale_top)
    bottom = int(config.char_size * config.use_bigger_canvas_scale_bottom)
    left = int(config.char_size * config.use_bigger_canvas_scale_left)
    right = int(config.char_size * config.use_bigger_canvas_scale_right)

    if shrink != 1:
        top = int(shrink * (top + half))
        bottom = int(shrink * (bottom + half))
        left = int(shrink * (left + half))
        right = int(shrink * (right + half))

    border_type = cv2.BORDER_CONSTANT
    resized_np_char_img = cv2.copyMakeBorder(np_char_img,
                                             top, bottom, left, right,
                                             border_type,
                                             value = [0,0,0]
                                             )
    return cv2.resize(resized_np_char_img, (config.char_size, config.char_size))

if __name__ == '__main__':
    config = parse_args()
    config = json.load(open(config, 'r', encoding='utf-8'))
    pprint(config)
    config = config_manager(override_dict=config)

    set_config(config)

    handle = generate_text_lines_with_text_handle(config)
    handle.generate_book_page_with_text()

    print('Done')
