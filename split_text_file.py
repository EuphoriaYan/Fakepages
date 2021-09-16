import os
from queue import Queue

text_file = "raw_text/leishu_sample.txt"

with open(text_file, 'r', encoding='utf-8') as fp:
    text = [line for line in fp]
    text = list(filter(None, text))
text = ''.join(text)

i = 0
# char_in_1MB = 341394
char_in_1MB = 20000

new_file_size = 1
new_text = Queue()

dict_str = text_file.split('/')
path = text_file[:-4]

isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

while len(text) != 0:
    l = min(char_in_1MB * new_file_size, len(text))
    splited_text = text[:l]
    name = path + '/' + dict_str[-1][:-4] + '_' + str(i) + '.txt'
    new_file = open(name, 'w', encoding='utf-8')
    new_file.write(splited_text)
    i += 1
    text = text[l:]


# for char in text:
#     new_text.put(char)
#     i += 1
#     if i >= new_file_size * char_in_1MB:
#         i = 0
