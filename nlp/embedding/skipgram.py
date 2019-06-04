# _*_ encoding: utf-8 _*_
import os
import fasttext


class FastTextSkipGramModel:

    def __init__(self, train_file,
                 model_path,
                 user_dict=None,
                 stop_dict=None):
        """
        使用fasttext训练词向量（skip-gram方式）
        :param train_file: 分好词的文本
        :param model_path: 模型保存的路劲
        :param user_dict: 自定义词典
        :param stop_dict: 停用词表
        """
        self.train_file = train_file
        self.model_path = model_path
        self.user_dict = user_dict
        self.stop_dict = stop_dict
        self.model = self.load()
        if not self.model:
            self.model = self.train()

    # 训练模型
    def train(self):
        # silent设为False, 训练过程会打印日志信息
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
