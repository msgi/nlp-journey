# _*_ encoding: utf-8 _*_
import fasttext
import jieba
import os


class FastTextSkipGramModel:

    def __init__(self, train_file,
                 model_path,
                 user_dict=None,
                 stop_dict=None):
        self.train_file = train_file
        self.model_path = model_path
        self.user_dict = user_dict
        self.stop_dict = stop_dict
        self.model = self.load()
        if not self.model:
            self.model = self.train()

    # 训练模型
    def train(self):
        return fasttext.skipgram(self.train_file, self.model_path, silent=False)

    # 返回词的向量
    def vector(self, word):
        return self.model[word]

    # 加载训练好的模型
    def load(self):
        if os.path.exists(self.model_path + 'bin'):
            return fasttext.load_model(self.model_path)
        else:
            return None


# 预处理数据，分词并去除停用词
def process_data(file, out_file, user_dict=None, stop_dict=None):
    if user_dict:
        jieba.load_userdict(user_dict)

    stop_words = []
    if stop_dict:
        with open(stop_dict, 'r', encoding='utf-8') as s:
            stop_words = s.readlines()

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [jieba.lcut(line.strip()) for line in lines]
        lines = [' '.join([l for l in line if l not in stop_words]) for line in lines]

    with open(out_file, 'w', encoding='utf-8') as o:
        o.writelines(lines)
