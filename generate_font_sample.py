# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import argparse
import numpy as np
import json
import random
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torchvision import transforms

from fontmagic_model import FontMagicModel


# Create bad_fonts_list
from util import processGlyphNames, draw_single_char, reverse_color


def get_bad_fontlist(create_num, bad_fonts='charset/error_font.txt'):
    with open(bad_fonts, 'r', encoding='utf-8') as bd_fs:
        bd_fs_lines = bd_fs.readlines()
    if create_num > 4:
        raise ValueError
    bd_fs_list = [int(num) for num in bd_fs_lines[create_num].strip().split()]
    return bd_fs_list


def get_fonts(fonts_json):
    dst_json = fonts_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)
    return dst_fonts


def chkormkdir(path):
    if os.path.isdir(path):
        return
    else:
        os.mkdir(path)
        return


class Icreate_char_handle:

    def update(self):
        raise ValueError

    def get_character(self, ch):
        raise ValueError


class create_mix_ch_handle(Icreate_char_handle):

    def __init__(self, bad_font_file, experiment_dir,
                 src_fonts_dir='charset/ZhongHuaSong',
                 fonts_json='/disks/sdb/projs/AncientBooks/data/chars/font_missing.json', fonts_root=None,
                 type_fonts='type/宋黑类字符集.txt',
                 input_nc=1, embedding_num=250, embedding_dim=128,  # model settings
                 Lconst_penalty=15, Lcategory_penalty=1.0, gpu_ids=['cuda'], resume=240000,  # model settings
                 char_size=250, canvas_size=256,
                 fake_prob=0.03):
        fontPlane00 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'))
        fontPlane02 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'))

        self.charSetPlane00 = processGlyphNames(fontPlane00.getGlyphNames())
        self.charSetPlane02 = processGlyphNames(fontPlane02.getGlyphNames())
        self.charSetTotal = self.charSetPlane00 | self.charSetPlane02
        self.charListTotal = list(self.charSetTotal)

        self.char_size = char_size
        self.canvas_size = canvas_size
        self.fake_prob = fake_prob

        self.fontPlane00 = ImageFont.truetype(
            os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), char_size)
        self.fontPlane02 = ImageFont.truetype(
            os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), char_size)

        self.fonts = self.get_fonts(fonts_json)
        self.fonts_root = fonts_root
        self.fonts2idx = {os.path.splitext(font['font_name'])[0]: idx for idx, font in enumerate(self.fonts)}

        with open(type_fonts, 'r', encoding='utf-8') as fp:
            self.type_fonts = {idx: font_line.strip() for idx, font_line in enumerate(fp)}
        self.type_fonts_rev = {v: k for k, v in self.type_fonts.items()}

        if bad_font_file:
            with open(bad_font_file, 'r', encoding='utf-8') as fp:
                self.bad_font_ids = [int(_) for _ in fp.readline().strip().split()]
        else:
            self.bad_font_ids = []
        self.fake_prob = 0.05

        checkpoint_dir = os.path.join(experiment_dir, "checkpoint")

        self.model = FontMagicModel(
            input_nc=input_nc,
            embedding_num=embedding_num,
            embedding_dim=embedding_dim,
            Lconst_penalty=Lconst_penalty,
            Lcategory_penalty=Lcategory_penalty,
            save_dir=checkpoint_dir,
            gpu_ids=gpu_ids,
            is_training=False
        )
        self.model.setup()
        self.model.print_networks(True)
        self.model.load_networks(resume)

    def update(self):
        self.set_cur_font()

    def get_fonts(self, fonts_json):
        dst_json = fonts_json
        with open(dst_json, 'r', encoding='utf-8') as fp:
            dst_fonts = json.load(fp)
        return dst_fonts

    def set_cur_font(self):
        self.idx = random.choice(list(self.type_fonts.keys()))
        cur_font_name = self.type_fonts[self.idx]
        raw_idx = self.fonts2idx[cur_font_name]

        cur_font = self.fonts[raw_idx]
        self.font_name = cur_font['font_name']
        print(self.font_name + ': ' + str(self.idx) + ' , raw_idx: ' + str(raw_idx), flush=True)

        if self.fonts_root is None:
            font_path = cur_font['font_pth']
        else:
            raw_font_path = cur_font['font_pth']
            font_basename = os.path.basename(raw_font_path)
            font_path = os.path.join(self.fonts_root, font_basename)
        self.font_missing = set(cur_font['missing'])
        self.font_fake = set(cur_font['fake'])
        self.dst_font = ImageFont.truetype(font_path, self.char_size)
        self.dst_font_chars = set(processGlyphNames(TTFont(font_path).getGlyphNames()))

    def get_fake_single_char(self, ch):
        if ch in self.charSetPlane00:
            input_img = draw_single_char(ch, self.fontPlane00, self.canvas_size)
        elif ch in self.charSetPlane02:
            input_img = draw_single_char(ch, self.fontPlane02, self.canvas_size)
        else:
            return None
        input_img = input_img.convert('L')
        input_tensor = transforms.ToTensor()(input_img)
        input_tensor = transforms.Normalize(0.5, 0.5)(input_tensor).unsqueeze(0)
        label = torch.tensor(self.idx, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            self.model.set_input(label, input_tensor, input_tensor)
            self.model.forward()
            output_tensor = self.model.fake_B.detach().cpu().squeeze(dim=0)
            img = transforms.ToPILImage('L')(output_tensor)
            return img

    def get_character(self, ch):
        # can draw this ch
        if ch in self.dst_font_chars and ch not in self.font_fake:
            if self.idx in self.bad_font_ids or random.random() > self.fake_prob:
                img = draw_single_char(ch, self.dst_font, self.canvas_size)
                if img is not None:
                    img = reverse_color(img)
                return img, True
            else:
                img = self.get_fake_single_char(ch)
                if img is not None:
                    img = reverse_color(img)
                return img, False
        # can't draw this ch
        else:
            # bad font, can't use font magic
            if self.idx in self.bad_font_ids:
                return None, True
            else:
                img = self.get_fake_single_char(ch)
                if img is not None:
                    img = reverse_color(img)
                return img, False


class create_ttf_ch_handle(Icreate_char_handle):

    def __init__(self, ttf_path, default_ttf_path, char_size, canvas_size):
        self.ttf_path_list = []
        for ttf_file in os.listdir(ttf_path):
            if os.path.splitext(ttf_file)[-1].lower() in ['.ttf', '.otf', '.ttc']:
                self.ttf_path_list.append(os.path.join(ttf_path, ttf_file))

        self.default_ttf_path = []
        for ttf_file in os.listdir(default_ttf_path):
            if os.path.splitext(ttf_file)[-1].lower() in ['.ttf', '.otf', '.ttc']:
                self.default_ttf_path.append(os.path.join(default_ttf_path, ttf_file))
        self.default_ttf_charset = []
        for default_ttf in self.default_ttf_path:
            ttfont = TTFont(default_ttf)
            self.default_ttf_charset.append(processGlyphNames(ttfont.getGlyphNames()))
        self.char_size = char_size
        self.canvas_size = canvas_size

    def update(self):
        self.current_font = random.choice(self.ttf_path_list)
        self.current_charset = processGlyphNames(TTFont(self.current_font).getGlyphNames())
        self.ttf_draw = ImageFont.truetype(self.current_font, self.char_size)

    def get_character(self, ch):
        if ch in self.current_charset:
            img = draw_single_char(ch, self.ttf_draw, self.canvas_size)
            if img is not None:
                img = reverse_color(img)
            return img, True
        else:
            for ttf_path, ttf_charset in zip(self.default_ttf_path, self.default_ttf_charset):
                if ch in ttf_charset:
                    default_draw = ImageFont.truetype(ttf_path, self.char_size)
                    img = draw_single_char(ch, default_draw, self.canvas_size)
                    if img is not None:
                        img = reverse_color(img)
                    return img, True
            return None, True


class create_imgs_ch_handle(Icreate_char_handle):
    # TODO use imgs to create fake pages
    def __init__(self, path):
        self.path = path

    def update(self):
        pass

    def get_character(self, ch):
        pass
