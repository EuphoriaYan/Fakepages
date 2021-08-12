from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

file = open("data/book_pages/book_pages_tags_vertical.txt", "r")
for line in file:
    print(line)
    list_str = line.replace('book_page_0.jpg\t', '')
    print(list_str)
    x = eval(list_str)
    print(x)
    text_bbox_list = x['text_bbox_list']
    char_bbox_list = x['char_bbox_list']
    print(text_bbox_list)

    img = Image.open('data/book_pages/imgs_vertical/book_page_0.jpg')
    draw = ImageDraw.Draw(img)
    # for bbox in text_bbox_list:
    #     bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    #     draw.rectangle([(bbox_x1, bbox_y1), (bbox_x2, bbox_y2)], fill=None, outline="red", width=6)
    for bbox in char_bbox_list:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        draw.rectangle([(bbox_x1, bbox_y1), (bbox_x2, bbox_y2)], fill=None, outline="blue", width=6)

    img.save('data/book_pages/imgs_vertical/book_page_0_rect_text_page.jpg')
