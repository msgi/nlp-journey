# _*_ encoding: utf-8 _*_
import os
import fasttext


class FastTextModel:

    def __init__(self, train_file,
                 model_path,
                 model_type='skipgram'):
        """
        用facebook的fasttext训练词向量（默认skipgram方式, 如果采用cbow方式，model_type设为'cbow'）
        :param train_file: 训练的文本，文件内容是分好词的
        :param model_path: 要存储的模型路径
        """
        self.train_file = train_file
        self.model_path = model_path
        self.model_type = model_type
        self.model = self.load()
        if not self.model:
            self.model = self.train()

    # 训练模型
    def train(self):
        if self.model_type == 'cbow':
            return fasttext.cbow(self.train_file, self.model_path)
        else:
            return fasttext.skipgram(self.train_file, self.model_path)

    # 返回词的向量
    def vector(self, word):
        return self.model[word]

    # 加载训练好的模型
    def load(self):
        if os.path.exists(self.model_path + 'bin'):
            return fasttext.load_model(self.model_path)
        else:
            return None
