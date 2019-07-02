# coding: utf-8
import jieba
from gensim.models import word2vec
from nlp.utils.pre_process import process_data


class GensimWord2VecModel:

    def __init__(self, train_file, model_path):
        """
        用gensim word2vec 训练词向量
        :param train_file: 分好词的文本
        :param model_path: 模型保存的路劲
        """
        self.train_file = train_file
        self.model_path = model_path
        self.model = self.load()
        if not self.model:
            self.model = self.train()
            self.save(self.model_path)

    def train(self):
        sentences = process_data(self.train_file)
        model = word2vec.Word2Vec(sentences, min_count=2, window=3, size=300, workers=4)
        return model

    def vector(self, word):
        return self.model.wv.get_vector(word)

    def similar(self, word):
        return self.model.wv.similar_by_word(word, topn=10)

    def save(self, model_path):
        self.model.save(model_path)

    def load(self):
        # 加载模型文件
        try:
            model = word2vec.Word2Vec.load(self.model_path)
        except FileNotFoundError:
            model = None
        return model