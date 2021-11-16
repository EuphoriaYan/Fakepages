# -*- coding: utf-8 -*-
# @Time   : 2021/9/16 07:53
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : generate_charset_xl.py

with open('charset/charset_xl.txt', 'r') as inf:
    content = inf.read()
content = content.replace('\n', '')
with open('raw_text/charset_xl.txt', 'w') as of:
    of.write(content)