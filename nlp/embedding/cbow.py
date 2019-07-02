# _*_ encoding: utf-8 _*_
import os
import fasttext


class FastTextCBowModel:

    def __init__(self, train_file, model_path):
        """
        用facebook的fasttext训练词向量（cbow方式）
        :param train_file: 训练的文本，文件内容是分好词的
        :param model_path: 要存储的模型路径
        """
        self.train_file = train_file
        self.model_path = model_path
        self.model = self.load()
        if not self.model:
            self.model = self.train()

    # 训练模型
    def train(self):
        return fasttext.cbow(self.train_file, self.model_path)

    # 返回词的向量
    def vector(self, word):
        return self.model[word]

    # 加载训练好的模型
    def load(self):
        if os.path.exists(self.model_path + 'bin'):
            return fasttext.load_model(self.model_path)
        else:
            return None
