from fontTools.ttLib import TTFont

from config import FONT_FILE_SEAL_DIR

from ocrodeg import *

def gen_font_labels_all(name_font, pth_font):
    font = TTFont(pth_font)
    unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
    out = ''
    for k,v in tqdm(unicode_map.items()):
        # 过滤
        uni_lst = ['uni','u1','u2']
        if not any(uni in v for uni in uni_lst) or len(v) < 6 : continue
        out += ('{} ----> {}\t{}\n'.format(k, v, chr(int(k))))
        # 不过滤
        # out += ('{}\t{}\t{}\n'.format(k,v,chr(int(k))))
    pth_font_labels = os.path.join('D:/GitHub/Fakepages/seal_labels', name_font+'.labels.txt')
    with open(pth_font_labels, 'w+', encoding='utf-8') as f:
        f.write(out)

if __name__ == '__main__':
    ttf_path_list = []
    for ttf_file in os.listdir(FONT_FILE_SEAL_DIR):
        try:
            gen_font_labels_all(ttf_file, os.path.join(FONT_FILE_SEAL_DIR, ttf_file))
        except:
            print(ttf_file)

    print('Done')