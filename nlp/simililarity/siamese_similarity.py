import itertools
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from nlp.utils.basic_log import Log
from nlp.utils.distances import exponent_neg_manhattan_distance
import logging

log = Log(logging.INFO)


def text_to_word_list(text):
    text = str(text)
    text = text.lower()
    # 清理
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
                 epochs=1,
                 train=False,
                 data_path=None,
                 embedding_file=None):
        self.model_path = model_path
        self.config_path = config_path
        # 加载停用词
        self.stops = set(stopwords.words('english'))
        if not train:
            self.embeddings, self.vocabulary, self.max_seq_length = self.__load_config()
            self.model = self.__load_model()
        else:
            self.data_path = data_path
            self.n_hidden = n_hidden
            self.gradient_clipping_norm = gradient_clipping_norm
            self.batch_size = batch_size
            self.epochs = epochs
            self.embedding_file = embedding_file
            self.x_train, self.y_train, self.x_validation, self.y_validation, self.vocabulary, self.max_seq_length = self.__load_data()
            self.embeddings = self.__load_word2vec(self.vocabulary)
            self.embedding_dim = self.embeddings.shape[1]
            self.model = self.train()

    def __build_model(self):
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')
        embedding_layer = Embedding(len(self.embeddings),
                                    self.embedding_dim,
                                    weights=[self.embeddings],
                                    input_length=self.max_seq_length,
                                    trainable=False)
        # Embedding
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        # 相同的lstm网络
        shared_lstm = LSTM(self.n_hidden)
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
                                  validation_data=(
                                      [self.x_validation['left'], self.x_validation['right']], self.y_validation))
        model.save(self.model_path)
        self.__save_config()
        self.__plot(model_trained)
        return model

    def __save_config(self):
        with open(self.config_path, 'wb') as out:
            pickle.dump((self.embeddings, self.vocabulary, self.max_seq_length), out)

    def predict(self, text1, text2):
        x1 = [self.vocabulary[word] for word in text_to_word_list(text1) if self.vocabulary[word] is not None]
        x2 = [self.vocabulary[word] for word in text_to_word_list(text2) if self.vocabulary[word] is not None]
        x1 = pad_sequences(x1, maxlen=self.max_seq_length)
        x2 = pad_sequences(x2, maxlen=self.max_seq_length)
        return self.model.predict([x1, x2])

    def __load_model(self):
        try:
            model = load_model(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def __load_word2vec(self, vocabulary):
        log.info('加载词向量...')
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(vocabulary) + 1, self.embedding_dim)
        embeddings[0] = 0

        for word, index in vocabulary.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    def __load_config(self):
        log.info('加载配置文件（词向量和最大长度）')
        with open(self.config_path, 'rb') as config:
            embeddings, vocabulary, max_seq_length = pickle.load(config)
        return embeddings, vocabulary, max_seq_length

    def __load_data(self):
        log.info('数据预处理...')
        vocabulary = dict()
        inverse_vocabulary = ['<unk>']
        questions_cols = ['question1', 'question2']

        log.info('加载数据集...')
        train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))

        # 找到最大的句子长度
        sentences = [df[col].str.split(' ') for df in [train_df, test_df] for col in questions_cols]
        max_seq_length = max([len(s) for ss in sentences for s in ss if isinstance(s, list)])
        # 预处理
        for dataset in [train_df, test_df]:
            for index, row in dataset.iterrows():
                for question in questions_cols:
                    q2n = []
                    for word in text_to_word_list(row[question]):
                        if word in self.stops:
                            continue
                        if word not in vocabulary:
                            vocabulary[word] = len(inverse_vocabulary)
                            q2n.append(len(inverse_vocabulary))
                            inverse_vocabulary.append(word)
                        else:
                            q2n.append(vocabulary[word])
                    dataset.set_value(index, question, q2n)

        validation_size = 40000
        x = train_df[questions_cols]
        y = train_df['is_duplicate']
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size)

        x_train = {'left': x_train.question1, 'right': x_train.question2}
        x_validation = {'left': x_validation.question1, 'right': x_validation.question2}

        y_train = y_train.values
        y_validation = y_validation.values

        for dataset, side in itertools.product([x_train, x_validation], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

        # 校验问题对各自数目是否正确
        assert x_train['left'].shape == x_train['right'].shape
        assert len(x_train['left']) == len(y_train)
        return x_train, y_train, x_validation, y_validation, vocabulary, max_seq_length

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
