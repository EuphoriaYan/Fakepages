# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import shutil
import pandas as pd

from config import CHINESE_LABEL_FILE_S, CHINESE_LABEL_FILE_ML, TRADITION_CHARS_FILE
from config import IGNORABLE_CHARS_FILE, IMPORTANT_CHARS_FILE

import torch
import torch.nn as nn
from torch.nn import init


def check_or_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def remove_then_makedirs(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def chinese_labels_dict(charset_size='m'):
    assert os.path.exists(CHINESE_LABEL_FILE_S), "Charset file does not exist!"
    assert os.path.exists(CHINESE_LABEL_FILE_ML), "Charset file does not exits!"

    if charset_size == 's':
        with open(CHINESE_LABEL_FILE_S, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        lines = [line.strip() for line in lines]
        id2char_dict = {i: k for i, k in enumerate(lines)}
        char2id_dict = {k: i for i, k in enumerate(lines)}
        num_chars = len(char2id_dict)
    elif charset_size == 'm':
        charset_csv = pd.read_csv(CHINESE_LABEL_FILE_ML)
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
        id2char_dict = {i: k for i, k in enumerate(charset)}
        char2id_dict = {k: i for i, k in enumerate(charset)}
        num_chars = len(char2id_dict)
    elif charset_size == 'l':
        charset_csv = pd.read_csv(CHINESE_LABEL_FILE_ML)
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.9999].values.squeeze(axis=-1).tolist()
        charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
        id2char_dict = {i: k for i, k in enumerate(charset)}
        char2id_dict = {k: i for i, k in enumerate(charset)}
        num_chars = len(char2id_dict)
    else:
        raise ValueError('charset_size should be s, m or l.')

    return id2char_dict, char2id_dict, num_chars


def ignorable_chars():
    chars = set()
    with open(IGNORABLE_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return chars


def important_chars():
    chars = set()
    with open(IMPORTANT_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return chars


# General tasks
ID2CHAR_DICT, CHAR2ID_DICT, NUM_CHARS = chinese_labels_dict()
BLANK_CHAR = ID2CHAR_DICT[0]
IGNORABLE_CHARS = ignorable_chars()
IMPORTANT_CHARS = important_chars()


def traditional_chars():
    with open(TRADITION_CHARS_FILE, "r", encoding="utf-8") as fr:
        tradition_chars = fr.read()
        tradition_chars = tradition_chars.strip()
    tradition_chars = "".join([c for c in tradition_chars if c in CHAR2ID_DICT])
    return tradition_chars


TRADITION_CHARS = traditional_chars()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if gpu_ids:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


if __name__ == '__main__':
    # print(ignorable_chars())
    print("Done !")
