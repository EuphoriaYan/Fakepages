
import os
import sys

from shutil import copy
import json
import random
import argparse


def convert_LTRB_to_poly(bbox):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]
    return [left, top, right, top, right, bottom, left, bottom]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_path', type=str)
    parser.add_argument('gt_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    imgs_path = args.imgs_path
    gt_path = args.gt_path

    if imgs_path.endswith('/') or imgs_path.endswith('\\'):
        imgs_path = imgs_path[:-1]

    (imgs_root, _) = os.path.split(imgs_path)
    print('image root: {}'.format(imgs_root))

    train_imgs_path = os.path.join(imgs_root, 'train_images')
    train_gts_path = os.path.join(imgs_root, 'train_gts')
    os.makedirs(train_imgs_path, exist_ok=True)
    os.makedirs(train_gts_path, exist_ok=True)
    print('train imgs path: {}'.format(train_imgs_path))
    print('train gts path: {}'.format(train_gts_path))

    test_imgs_path = os.path.join(imgs_root, 'test_images')
    test_gts_path = os.path.join(imgs_root, 'test_gts')
    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_gts_path, exist_ok=True)
    print('test imgs path: {}'.format(test_imgs_path))
    print('test gts path: {}'.format(test_gts_path))

    train_imgs_list_path = os.path.join(imgs_root, 'train_list.txt')
    test_imgs_list_path = os.path.join(imgs_root, 'test_list.txt')

    total_timgs = os.listdir(imgs_path)
    total_cnt = len(total_timgs)

    test_imgs_list = random.choices(total_timgs, k=total_cnt // 50 )
    test_imgs_set = set(test_imgs_list)
    train_imgs_list = [img for img in total_timgs if img not in test_imgs_set]
    train_imgs_set = set(train_imgs_list)

    with open(train_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in train_imgs_list:
            fp.write(timg + '\n')

    with open(test_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in test_imgs_list:
            fp.write(timg + '\n')

    with open(gt_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            timg, img_json = line.strip().split('\t')
            timg_real_path = os.path.join(imgs_path, timg)
            img_json = json.loads(img_json)
            bboxs = img_json['text_bbox_list']
            texts = img_json['text_list']
            records = []
            for bbox, text in zip(bboxs, texts):
                poly = convert_LTRB_to_poly(bbox)
                poly = list(map(str, poly))
                text = ''.join(text)
                records.append(','.join(poly + [text]))
            if timg in train_imgs_set:
                gt_path = os.path.join(train_gts_path, timg + '.txt')
                dst_path = os.path.join(train_imgs_path, timg)
            elif timg in test_imgs_set:
                gt_path = os.path.join(test_gts_path, timg + '.txt')
                dst_path = os.path.join(test_imgs_path, timg)
            else:
                raise ValueError
            with open(gt_path, 'w', encoding='utf-8') as fp:
                for record in records:
                    fp.write(record + '\n')
            copy(timg_real_path, dst_path)


