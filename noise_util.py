# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import random
import math
import json
from img_utils import reverse_image_color



def cal_dis(pA, pB):
    return math.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)


def add_noise(img, generate_ratio=0.003, generate_size=0.006):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R_max = max(3, int(min(h, w) * generate_size))
    threshold = int(h * w * generate_ratio)
    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))
    cnt = 0

    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(R, w - 1 - R)
        P_noise_y = random.randint(R, h - 1 - R)
        for i in range(P_noise_x - R, P_noise_x + R):
            for j in range(P_noise_y - R, P_noise_y + R):
                if cal_dis((i, j), (P_noise_x, P_noise_y)) < random.randint(int(R/2), R):
                    if random.random() < 0.6:
                        img[j][i] = random.randint(0, 255)
        cnt += 2 * R
        if cnt >= threshold:
            break

    R_max *= 2
    random_choice_list = []
    for i in range(1, R_max + 1):
        random_choice_list.extend([i] * (R_max - i + 1))
    cnt = 0
    while True:
        R = random.choice(random_choice_list)
        P_noise_x = random.randint(0, w - 1 - R)
        P_noise_y = random.randint(0, h - 1 - R)
        for i in range(P_noise_x + 1, P_noise_x + R):
            for j in range(P_noise_y + 1, P_noise_y + R):
                if cal_dis((i, j), (P_noise_x + int(R/2), P_noise_y + int(R/2))) < random.randint(int(R/4), int(R*0.7)):
                    if random.random() < 0.6:
                        img[j][i] = random.randint(0, 255)
        cnt += R
        if cnt >= threshold:
            break

    img = Image.fromarray(img)
    return img

def white_erosion(img, generate_ratio=0.01, generate_size=0.04):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    h, w = img.shape
    R_max = max(10, int(min(h, w) * generate_size))

    random_choise_list = []
    for i in range(R_max, h - R_max):
        for j in range(R_max, w - R_max):
            if img[i, j] > 150:  # 找附近有黑色像素的白色像素点
                if img[i+1, j] < 150 or img[i-1, j] < 150 or img[i, j+1] < 150 or img[i, j-1] < 150 :
                    random_choise_list.append([i, j])

    threshold = int(w * h * (5 * (w+h) / len(random_choise_list)) * generate_ratio)

    cnt = 0
    while True:
        x, y = random.choice(random_choise_list)
        R = random.randint(5, R_max)
        for i in range(x - R, x + R):
            for j in range(y - R, y + R):  # 以这个点为圆心，一半半径羽化边缘随机增加白色像素噪音
                if cal_dis((i, j), (x, y)) < random.randint(int(R/2), R):
                    if random.random() < 0.6:
                        img[i][j] = 255
        cnt += 2 * R
        if cnt >= threshold:
            break

    img = Image.fromarray(img)
    return img

def triangle_contrast(PIL_page):
    img = np.array(PIL_page, dtype=np.uint8)
    img = img/255

    rows, cols = img.shape

    blank = np.zeros([rows, cols], img.dtype)

    a_x = random.randint(0, cols)
    b_x = random.randint(0, cols)
    c_x = random.randint(0, cols)

    a_y = random.randint(0, rows)
    b_y = random.randint(0, rows)
    c_y = random.randint(0, rows)

    triangle = np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y]])

    c = random.uniform(0.02, 0.3)

    cv2.fillConvexPoly(blank, triangle, (c, c, c))

    dst = img + blank
    dst[dst > 1] = 1

    dst = dst*255

    PIL_page = Image.fromarray(dst.astype(np.uint8))
    return PIL_page