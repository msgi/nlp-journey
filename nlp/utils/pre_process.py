import jieba


# 预处理数据，分词并去除停用词，然后写回文件
def seg_to_file(file, out_file, user_dict=None, stop_dict=None):
    if user_dict:
        jieba.load_userdict(user_dict)
    stop_words = []
    if stop_dict:
        with open(stop_dict, 'r', encoding='utf-8') as s:
            stop_words = [stop_word.strip() for stop_word in s.readlines()]

    with open(file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [jieba.lcut(line) for line in lines]
        lines = [' '.join([l for l in line if l not in stop_words]) for line in lines]

    with open(out_file, 'w', encoding='utf-8') as o:
        o.writelines(lines)
