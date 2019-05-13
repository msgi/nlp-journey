import logging
import os
import time
import pickle
from collections import Counter
import jieba

from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GRU, BatchNormalization, CuDNNLSTM
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from nlp.layers.attention import Attention
from keras.datasets import imdb
import numpy as np
from nlp.utils.plot_model_history import plot
from nlp.utils.basic_log import Log

log = Log(logging.INFO)


class BiLSTMAttentionClassifier:

    def __init__(self, embedding_file,
                 model_path,
                 config_path,
                 train=False,
                 file_path=None,
                 embed_size=300):
        self.embed_size = embed_size
        self.embedding_file = embedding_file
        self.model_path = model_path
        self.config_path = config_path
        self.word_index, self.maxlen, self.embedding_matrix = self.__load_config()
        if not train:
            self.model = self.__load_model()
            assert self.model is not None, '模型导入失败'
        else:
            self.file_path = file_path
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.word_index, self.maxlen = self.__preprocess()
            # 加载词向量过程比较长，可以预先将需要的词向量写入配置文件中，训练的时候直接加载即可
            if self.embedding_matrix is None:
                self.embedding_matrix = self.__load_embedding()
            self.model = self.train()

    # 采用注意力机制
    def __build_model(self):
        inp = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embedding_matrix),
                      self.embed_size,
                      weights=[self.embedding_matrix],
                      trainable=False)(inp)
        x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
        x = Attention()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # 没有采用注意力机制
    def __build_model_no_attention(self):
        inp = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embedding_matrix),
                      self.embed_size,
                      weights=[self.embedding_matrix],
                      trainable=False)(inp)
        x = Bidirectional(CuDNNLSTM(150, dropout=0.25, recurrent_dropout=0.25))(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # 训练开始
    def train(self):
        model = self.__build_model()
        # model = self.__build_model_no_attention()
        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'weights.{epoch:03d}-{val_acc:.4f}.h5'),
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=10)
        model_trained = model.fit(self.x_train,
                                  self.y_train,
                                  batch_size=128,
                                  epochs=25,
                                  validation_data=[self.x_test, self.y_test],
                                  callbacks=[checkpoint, early])
        plot(model_trained)
        model.save_weights(os.path.join(self.model_path, 'final_model_weights.h5'))
        # model.save_weights(os.path.join(self.model_path, 'final_model_weights_no_attention.h5'))
        self.__save_config()
        return model

    # 使用keras内置的imdb数据集做基准训练
    @staticmethod
    def __preprocess():
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        word_index = imdb.get_word_index()

        maxlen = 500
        x_train = pad_sequences(train_data, maxlen=maxlen)
        x_test = pad_sequences(test_data, maxlen=maxlen)
        y_train = np.asarray(train_labels).astype('float32')
        y_test = np.asarray(test_labels).astype('float32')

        return (x_train, y_train), (x_test, y_test), word_index, maxlen

    # 用自己的数据集做训练,只支持二分类（格式：分好词的句子##标签，如：我 很 喜欢 这部 电影#pos）
    def __preprocess_raw(self, test_size=0.2):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split('##') for line in lines]
        x = [line[0] for line in lines]
        x = [line.split() for line in x]
        data = [word for xx in x for word in xx]
        y = [line[0] for line in lines]

        counter = Counter(data)
        vocab = [k for k, v in counter.items() if v >= 5]

        word_index = {k: v for v, k in enumerate(vocab)}
        maxlen = max([len(words) for words in x])

        x_data = [[word_index[word] for word in words if word in word_index] for words in x]
        y_data = [1 if x == 'pos' else 0 for x in y]

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
        return x_train, y_train, x_test, y_test, word_index, maxlen

    # 用训练好的词向量
    def __load_embedding(self):
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(self.word_index) + 1, self.embed_size)
        embeddings[0] = 0

        for word, index in self.word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    # 加载训练好的模型
    def __load_model(self):
        try:
            model = self.__build_model()
            model.load_weights(os.path.join(self.model_path, 'final_model_weights.h5'))
        except FileNotFoundError:
            model = None
        return model

    # 保存配置文件(训练阶段)
    def __save_config(self):
        with open(self.config_path, 'wb') as config:
            pickle.dump((self.word_index, self.maxlen, self.embedding_matrix), config)

    # 加载配置文件(纯推理阶段)
    def __load_config(self):
        with open(self.config_path, 'rb') as config:
            word_index, maxlen, embedding_matrix = pickle.load(config)
        return word_index, maxlen, embedding_matrix

    # 推理出数值
    def predict(self, text):
        if isinstance(text, list):
            x_predict = [self.word_index.get(x, 0) for t in text for x in t.split()]
        else:
            x_predict = [self.word_index.get(x, 0) for x in text.split()]
        print(x_predict)
        x_predict = pad_sequences([x_predict], maxlen=self.maxlen)
        return self.model.predict(x_predict)

    # 推理出结果
    def predict_result(self, texts):
        results = self.predict(texts)
        results = ['pos' if x > 0.5 else 'neg' for x in results]
        return results
