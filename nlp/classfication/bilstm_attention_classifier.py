import logging
import os
import time
import pickle

from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GRU
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

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
                 embed_size=300):
        self.embed_size = embed_size
        self.embedding_file = embedding_file
        self.model_path = model_path
        self.config_path = config_path

        if not train:
            self.word_index, self.maxlen = self.__load_config()
            self.model = self.__load_model()
            assert self.model is not None, '模型导入失败'
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.word_index, self.maxlen = self.__preprocess()
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
        x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
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
        checkpoint_dir = os.path.join(self.model_path, 'checkpoints/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + str(int(time.time())) + '.h5'
        checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        model_trained = model.fit(self.x_train,
                                  self.y_train,
                                  batch_size=256,
                                  epochs=25,
                                  validation_data=[self.x_test, self.y_test],
                                  callbacks=[checkpoint, early])
        plot(model_trained)
        model.save(os.path.join(self.model_path, 'final_model.h5'))
        return model

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

    def __preprocess_raw(self):
        pass

    def __load_embedding(self):
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(self.word_index) + 1, self.embed_size)
        embeddings[0] = 0

        for word, index in self.word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    def __load_model(self):
        try:
            model = load_model(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def __save_config(self):
        with open(self.config_path, 'w') as config:
            pickle.dump((self.word_index, self.maxlen), config)

    def __load_config(self):
        with open(self.config_path , 'r') as config:
            word_index, maxlen = pickle.load(config)
        return word_index, maxlen

    def predict(self, text):
        if isinstance(text, list):
            x_predict = [self.word_index.get(x, 0) for t in text for x in t.split()]
        else:
            x_predict = [self.word_index.get(x, 0) for x in text.split()]

        print(x_predict)

        x_predict = pad_sequences(x_predict,maxlen=self.maxlen)

        return self.model.predict(x_predict)
