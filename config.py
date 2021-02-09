# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os

# ************************ basic configuration ***************************
# os.getcwd() returns the current working directory
CURR_DIR = os.path.dirname(__file__)  # .replace("/", os.sep)
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
BOOK_PAGE_TAGS_FILE_V = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_vertical_3.txt")

FONT_FILE_DIR = os.path.join(CURR_DIR, "chinese_fonts")
FONT_FINISHED_DIR = os.path.join(CURR_DIR, "chinese_fonts_finished")

SHUFA_FILE_DIR = os.path.join(CURR_DIR, "shufa_imgs")

# From "Diaolong" dataset, collect different shapess
BOOK_PAGE_SHAPE_LIST = [(740, 990), (1480, 990), (2295, 1181), (2313, 1181),
                        (2387, 1206), (2390, 1243), (2439, 1184), (3168, 2382)]

EXTERNEL_IMAGES_DIR = "./ziku_images"
# ************************ generate image data ***************************

# chinese char images
CHAR_IMG_SIZE = 64
MAX_ROTATE_ANGLE = 5
NUM_IMAGES_PER_FONT = 10

# text line images
TEXT_LINE_SIZE = CHAR_IMG_SIZE

