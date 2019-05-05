import itertools
import logging
import os
import pickle
import re

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding, LSTM, Lambda, Bidirectional
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from nlp.utils.basic_log import Log

log = Log(logging.INFO)


# 曼哈顿距离
def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def text_to_list(text):
    text = str(text)
    text = text.lower()
    # 清理数据
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text


class SiameseSimilarity:
    def __init__(self,
                 model_path,
                 config_path,
                 n_hidden=50,
                 gradient_clipping_norm=1.25,
                 batch_size=64,
                 epochs=2,
                 train=False,
                 embedding_dim=300,
                 data_path=None,
                 embedding_file=None):
        """
        初始化
        :param model_path: 要保存的或者已经保存的模型路径
        :param config_path: 要保存的或者已经保存的配置文件路径
        :param n_hidden: lstm隐藏层维度
        :param gradient_clipping_norm: adadelta参数
        :param batch_size:
        :param epochs:
        :param train: 是否训练模式，如果是训练模式，则必须提供data_path
        :param data_path: 存放了train.csv和test.csv的目录
        :param embedding_file: 训练好的词向量文件
        """
        self.model_path = model_path
        self.config_path = config_path
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.gradient_clipping_norm = gradient_clipping_norm
        # 加载停用词
        self.stops = set(stopwords.words('english'))
        if not train:
            self.embeddings, self.word_index, self.max_length = self.__load_config()
            self.model = self.__load_model()
        else:
            assert data_path is not None, '训练模式，训练数据必须！'
            assert embedding_file is not None, '训练模式，训练好的词向量数据必须！'
            self.data_path = data_path
            self.batch_size = batch_size
            self.epochs = epochs
            self.embedding_file = embedding_file
            self.train_data = os.path.join(self.data_path, 'train.csv')
            self.test_data = os.path.join(self.data_path, 'test.csv')
            self.x_train, self.y_train, self.x_val, self.y_val, self.word_index, self.max_length = self.__load_data(
                self.train_data, self.test_data)
            self.embeddings = self.__load_word2vec(self.word_index)
            self.model = self.train()

    def __build_model(self):
        left_input = Input(shape=(self.max_length,), dtype='int32')
        right_input = Input(shape=(self.max_length,), dtype='int32')
        embedding_layer = Embedding(len(self.embeddings),
                                    self.embedding_dim,
                                    weights=[self.embeddings],
                                    input_length=self.max_length,
                                    trainable=False)
        # Embedding
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        # 相同的lstm网络
        shared_lstm = Bidirectional(LSTM(self.n_hidden // 2))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        # 计算距离
        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        # 构造模型
        model = Model([left_input, right_input], [malstm_distance])
        # Adadelta优化器
        optimizer = Adadelta(clipnorm=self.gradient_clipping_norm)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self):
        model = self.__build_model()
        model_trained = model.fit([self.x_train['left'], self.x_train['right']],
                                  self.y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  validation_data=([self.x_val['left'], self.x_val['right']], self.y_val),
                                  verbose=1)
        model.save_weights(self.model_path)
        self.__save_config()
        self.__plot(model_trained)
        return model

    def __save_config(self):
        with open(self.config_path, 'wb') as out:
            pickle.dump((self.embeddings, self.word_index, self.max_length), out)
        if out:
            out.close()

    # 推理两个文本的相似度，大于0.5则相似，否则不相似
    def predict(self, text1, text2):
        if isinstance(text1, list) or isinstance(text2,list):
            x1 = [[self.word_index.get(word, 0) for word in text_to_list(text)] for text in text1]
            x2 = [[self.word_index.get(word, 0) for word in text_to_list(text)] for text in text2]
            x1 = pad_sequences(x1, maxlen=self.max_length)
            x2 = pad_sequences(x2, maxlen=self.max_length)
        else:
            x1 = [self.word_index.get(word, 0) for word in text_to_list(text1)]
            x2 = [self.word_index.get(word, 0) for word in text_to_list(text2)]
            x1 = pad_sequences([x1], maxlen=self.max_length)
            x2 = pad_sequences([x2], maxlen=self.max_length)
        # 转为词向量
        return self.model.predict([x1, x2])

    def __load_model(self):
        try:
            model = self.__build_model()
            model.load_weights(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def __load_word2vec(self, word_index):
        log.info('加载词向量...')
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(word_index) + 1, self.embedding_dim)
        embeddings[0] = 0

        for word, index in word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    def __load_config(self):
        log.info('加载配置文件（词向量和最大长度）')
        with open(self.config_path, 'rb') as config:
            embeddings, vocabulary, max_seq_length = pickle.load(config)
        if config:
            config.close()
        return embeddings, vocabulary, max_seq_length

    def __load_data(self, train_csv, test_csv, test_size=0.2):
        log.info('数据预处理...')
        # word:index和index:word
        word_index = dict()
        index_word = ['<unk>']
        questions_cols = ['question1', 'question2']

        log.info('加载数据集...')
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # 找到最大的句子长度
        sentences = [df[col].str.split(' ') for df in [train_df, test_df] for col in questions_cols]
        max_seq_length = max([len(s) for ss in sentences for s in ss if isinstance(s, list)])
        # 预处理(统计并将字符串转换为索引)
        for dataset in [train_df, test_df]:
            for index, row in dataset.iterrows():
                for question_col in questions_cols:
                    question_indexes = []
                    for word in text_to_list(row[question_col]):
                        if word in self.stops:
                            continue
                        if word not in word_index:
                            word_index[word] = len(index_word)
                            question_indexes.append(len(index_word))
                            index_word.append(word)
                        else:
                            question_indexes.append(word_index[word])
                    dataset.set_value(index, question_col, question_indexes)

        x = train_df[questions_cols]
        y = train_df['is_duplicate']
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)

        x_train = {'left': x_train.question1, 'right': x_train.question2}
        x_val = {'left': x_val.question1, 'right': x_val.question2}

        y_train = y_train.values
        y_val = y_val.values

        for dataset, side in itertools.product([x_train, x_val], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

        # 校验问题对各自数目是否正确
        assert x_train['left'].shape == x_train['right'].shape
        assert len(x_train['left']) == len(y_train)
        return x_train, y_train, x_val, y_val, word_index, max_seq_length

    @staticmethod
    def __plot(model_trained):
        # Plot accuracy
        plt.plot(model_trained.history['acc'])
        plt.plot(model_trained.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot loss
        plt.plot(model_trained.history['loss'])
        plt.plot(model_trained.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
