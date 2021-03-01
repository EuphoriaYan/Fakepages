# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import copy
import os

# ************************ basic configuration ***************************
import random
import re
from queue import Queue

CURR_DIR = os.path.dirname(__file__)
CHINESE_LABEL_FILE_S = os.path.join(CURR_DIR, "chinese_labels", "charset_s.txt")
CHINESE_LABEL_FILE_ML = os.path.join(CURR_DIR, "chinese_labels", "all_abooks.unigrams_desc.Clean.rate.csv")
TRADITION_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "chars_Big5_all_traditional_13053.txt")
IGNORABLE_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "ignorable_chars.txt")
IMPORTANT_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "important_chars.txt")
# ************************ basic configuration ***************************


# ************************ generate image data ***************************
DATA_DIR = os.path.join(CURR_DIR, "data")

CHAR_IMGS_DIR = os.path.join(DATA_DIR, "chars", "imgs")

ONE_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "one_text_lines", "imgs_horizontal")
ONE_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "one_text_lines", "imgs_vertical")
ONE_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "one_text_lines", "text_lines_tags_horizontal.txt")
ONE_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "one_text_lines", "text_lines_tags_vertical.txt")

TWO_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "two_text_lines", "imgs_horizontal")
TWO_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "two_text_lines", "imgs_vertical")
TWO_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "two_text_lines", "text_lines_tags_horizontal.txt")
TWO_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "two_text_lines", "text_lines_tags_vertical.txt")

MIX_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "mix_text_lines", "imgs_horizontal")
MIX_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "mix_text_lines", "imgs_vertical")
MIX_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "mix_text_lines", "text_lines_tags_horizontal.txt")
MIX_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "mix_text_lines", "text_lines_tags_vertical.txt")

BOOK_PAGE_IMGS_H = os.path.join(DATA_DIR, "book_pages", "imgs_horizontal")
BOOK_PAGE_IMGS_V = os.path.join(DATA_DIR, "book_pages", "imgs_vertical")
BOOK_PAGE_TAGS_FILE_H = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_horizontal.txt")
BOOK_PAGE_TAGS_FILE_V = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_vertical.txt")

FONT_FILE_DIR = os.path.join(CURR_DIR, "chinese_fonts")
DEFAULT_FONT_FILE_DIR = os.path.join(CURR_DIR, "charset/ZhongHuaSong")
SHUFA_FILE_DIR = os.path.join(CURR_DIR, "shufa_imgs")

# From "Diaolong" dataset, collect different shapess
BOOK_PAGE_SHAPE_LIST = [(740, 990), (1480, 990), (2295, 1181), (2313, 1181),
                        (2387, 1206), (2390, 1243), (2439, 1184), (3168, 2382)]

EXTERNEL_IMAGES_DIR = "./ziku_images"
# ************************ generate image data ***************************


class config_manager:
    def __init__(self, override_dict=None):
        if override_dict is None:
            override_dict = dict()

        self.obj_num = 10  # 生成数量
        self.shape = BOOK_PAGE_SHAPE_LIST  # 页面形状（宽*高）
        self.text_from = 'text'  # 文本来源，[text|dict|random]，dict必须配合config_type='dict'
        # self.text = None  # 生成的文字
        self.text ='./raw_text/leishu_sample.txt'
        self.delete_symbol = True
        self.keep_line_break = False  # 是否保留\n
        self.orient = 'vertical'  # 生成的方向
        self.char_size = 64  # 单字生成时的大小，默认为64
        self.canvas_size = 64  # 画布大小，等于或微大于char_size即可，需要同步修改
        self.max_rotate_angle = 5  # 最大的旋转角度

        self.full_line = True  # 是否填满整行or整列
        self.line_length = 1  # 最小整行长度（占总行长的比例）

        self.limit_max_length_single = 20  # 单列or行的最长字数
        self.limit_max_length_double = 10  # 双列or行的最长字数
        self.limit_min_length_single = 1  # 单列or行的最少字数
        self.limit_min_length_double = 1  # 双列or行的最少字数

        self.start_at_single = 0.6  # 每行起始为单行的概率
        self.end_at_single = False  # 凡遇单行，则换行

        self.symbol_on_char = False  # 在字图上再加符号
        self.symbol_path = 'charset/symbol'  # 符号图的位置
        self.symbol_use = ['wave', 'reverse', 'underline']  # 使用哪些符号，详见下方dict
        self.symbol_prob = [0.1, 0.01, 0.1]  # 符号出现的概率，和symbol_use同样长度
        self.symbol_dict = {'underline': os.path.join(self.symbol_path, '下划线_细直线.PNG'),
                            'reverse': os.path.join(self.symbol_path, '印章.PNG'),
                            'wave': os.path.join(self.symbol_path, '下划线_波浪.PNG'),
                            'two_circles': os.path.join(self.symbol_path, 'two_circles.png'),
                            'one_circle_black': os.path.join(self.symbol_path, 'one_circle_black.png'),
                            'one_circle_white': os.path.join(self.symbol_path, 'one_circle_white.png'),
                            'big_circle': os.path.join(self.symbol_path, 'big_circle.png'),
                            'comma': os.path.join(self.symbol_path, '逗号.PNG'),
                            'period': os.path.join(self.symbol_path, '句号.PNG'),}

        self.symbol_next_char = False  # 不与字图重叠的符号
        self.symbol_next_use = ['big_circle']  # 使用哪些符号
        self.symbol_next_prob = [0.1]  # 符号出现的概率，和symbol_next_use同样长度
        self.symbol_next_dict ={}
        self.symbol_number_next_char = ['chinese_number_with_circle_reverse',  # 带序号的特殊符号
                                        'chinese_number_with_circle',
                                        'number_with_circle_reverse',
                                        'number_with_circle']

        self.char_reshape = False  # 将文字横向或纵向拉长（单行or列）
        self.char_reshape_line = 'single'  # 选择reshape的行or列类型 [single|double|both]
        self.char_single_line_reshape_stretch = 1  # 拉长的倍数（若文字纵向排列，则为文字高的倍数；横向为宽的倍数）
        self.char_double_line_reshape_stretch = 1  # 拉长的倍数（若文字纵向排列，则为文字高的倍数；横向为宽的倍数）

        self.use_bigger_canvas = False  # 将字图放在一个更大的画框里，用于永乐大典等宽松排版
        # self.use_bigger_canvas_scale = 2  # 画框的放大倍数
        self.use_bigger_canvas_scale_top = 0  # 画框的上沿
        self.use_bigger_canvas_scale_bottom = 0  # 画框的下沿
        self.use_bigger_canvas_scale_left = 0  # 画框的左沿
        self.use_bigger_canvas_scale_right = 0  # 画框的右沿


        self.augment = True  # 添加augment

        self.char_from = 'ttf'  # 单字来源，[fontmagic|ttf|imgs]
        self.fonts_json = None  # 生成的fonts_json文件
        self.fonts_root = None  # 如果fonts_json的path和真实的path不符，填入真实path的前缀
        self.bad_font_file = None  # fonts里哪些是不能使用font magic的
        self.font_magic_experiment = None  # font magic模型位置
        self.type_fonts = None  # 指示font magic里每个index分别是哪个字体
        self.embedding_num = None  # 指示font magic里一共有多少个字体
        self.resume = None  # font magic从第几步的模型进行恢复

        self.char_imgs_path = SHUFA_FILE_DIR  # 如果使用字图，字图在哪里

        self.ttf_path = FONT_FILE_DIR  # 如果使用ttf，ttf文件在哪里
        self.default_ttf_path = DEFAULT_FONT_FILE_DIR  # 如果使用ttf/字图且遇到不可绘制文字，默认的ttf文件

        self.line_type = 'mixed'  # 单行或混合单双行，或为字典特殊设计的行结构 [mixed|single|dict]
        self.line_num = (10, 15)

        self.charset_file = 'charset/charset_xl.txt'  # 字符集
        self.init_num = 0  # 生成的初始序号，用于中断后继续生成

        self.special_type = []  # 特殊的格式  从[split|num_end|num_start]中选择并加入
        self.segment_type = 'normal'  # 字之间间隔格式 [normal|spacious|crowded|mixed]

        self.store_imgs = BOOK_PAGE_IMGS_V   # 在哪里存储图片
        self.store_tags = BOOK_PAGE_TAGS_FILE_V  # 在哪里存储标签

        self.init_config(override_dict)

    def init_config(self, override_dict):
        for key, value in override_dict.items():
            setattr(self, key, value)
        self.init_text()
        with open(self.charset_file, 'r', encoding='utf-8') as charset_file:
            charset = set([s.strip() for s in charset_file.readlines()])
            self.charset = charset

    def init_text(self):
        if self.text_from == 'text':
            self.get_text_queue(self.text, self.keep_line_break)
        elif self.text_from == 'dict':
            pass  # TODO fix dict
        elif self.text_from == 'random':
            self.get_text_queue('cover_charset', self.keep_line_break)
        else:
            raise ValueError

    def get_text_queue(self, text_file, keep_line_break):
        if text_file == 'cover_charset':
            text = []
            with open('./charset/charset_xl.txt', 'r', encoding='utf-8') as fp:
                raw_charset = [line.strip() for line in fp]
            for _ in range(50):
                charset = copy.deepcopy(raw_charset)
                random.shuffle(charset)
                text.extend(charset)
            text = ''.join(text)
        else:
            with open(text_file, 'r', encoding='utf-8') as fp:
                if not keep_line_break:
                    text = [line.strip() for line in fp]
                text = [re.sub('[，。“”‘’？！《》、（）〔〕:：；;·［］【】〈〉<>︻︼︵︶︹︺△　]', '', line) for line in fp]
                text = list(filter(None, text))
            text = ''.join(text)
        self.text = Queue()
        for char in text:
            self.text.put(char)

