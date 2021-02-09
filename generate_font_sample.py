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


def processGlyphNames(GlyphNames):
    res = set()
    for char in GlyphNames:
        if char.startswith('uni'):
            char = char[3:]
        elif char.startswith('u'):
            char = char[1:]
        else:
            continue
        if char:
            try:
                char_int = int(char, base=16)
            except ValueError:
                continue
            try:
                char = chr(char_int)
            except ValueError:
                continue
            res.add(char)
    return res


def draw_single_char(ch, font, canvas_size):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


# Create bad_fonts_list
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


class create_mix_ch_handle:

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

    def reverse_color(self, PIL_image):
        img = np.array(PIL_image)
        img = 255 - img
        img = Image.fromarray(img)
        return img

    def get_mix_character(self, ch):
        # can draw this ch
        if ch in self.dst_font_chars and ch not in self.font_fake:
            if self.idx in self.bad_font_ids or random.random() > self.fake_prob:
                img = draw_single_char(ch, self.dst_font, self.canvas_size)
                if img is not None:
                    img = self.reverse_color(img)
                return img, True
            else:
                img = self.get_fake_single_char(ch)
                if img is not None:
                    img = self.reverse_color(img)
                return img, False
        # can't draw this ch
        else:
            # bad font, can't use font magic
            if self.idx in self.bad_font_ids:
                return None, True
            else:
                img = self.get_fake_single_char(ch)
                if img is not None:
                    img = self.reverse_color(img)
                return img, False
