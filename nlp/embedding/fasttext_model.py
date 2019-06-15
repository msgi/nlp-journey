# _*_ encoding: utf-8 _*_
import os
import fasttext


class FastTextModel:

    def __init__(self, train_file,
                 model_path,
                 model_type='skipgram',
                 user_dict=None,
                 stop_dict=None):
        """
        用facebook的fasttext训练词向量（cbow方式）
        :param train_file: 训练的文本，文件内容是分好词的
        :param model_path: 要存储的模型路径
        :param user_dict: 用户自定义词典
        :param stop_dict: 停用词典
        """
        self.train_file = train_file
        self.model_path = model_path
        self.model_type = model_type
        self.user_dict = user_dict
        self.stop_dict = stop_dict
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
