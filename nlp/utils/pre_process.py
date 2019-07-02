import jieba


# 预处理数据，分词并去除停用词，然后写回文件
def seg_to_file(file, out_file, user_dict=None, stop_dict=None):
    sentences = process_data(file, user_dict,stop_dict)
    sentences = [' '.join([l for l in line if l not in stop_words]) for line in lines]

    with open(out_file, 'w', encoding='utf-8') as o:
        o.writelines(lines)


def process_data(train_file, user_dict=None, stop_dict=None):
    # 结巴分词加载自定义词典(要符合jieba自定义词典规范)
    if user_dict:
        jieba.load_userdict(user_dict)

    # 加载停用词表(每行一个停用词)
    stop_words = []
    if stop_dict:
        with open(stop_dict, 'r', encoding='utf-8') as file:
            stop_words = [stop_word.strip() for stop_word in file.readlines()]

    # 读取文件内容并分词, 去掉停用词
    with open(train_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
        sentences = [jieba.lcut(sentence.strip()) for sentence in sentences]
        sentences = [[s for s in sentence if s not in stop_words] for sentence in sentences]

    return sentences