import logging
import os
import time

from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GRU
from keras_preprocessing.sequence import pad_sequences

from nlp.layers.attention import Attention
from keras.datasets import imdb
import numpy as np
from nlp.utils.plot_model_history import plot
from nlp.utils.basic_log import Log

log = Log(logging.INFO)


class BiLSTMAttentionModel:

    def __init__(self, embedding_file,
                 model_path,
                 embed_size=300):
        self.embed_size = embed_size
        self.embedding_file = embedding_file
        self.model_path = model_path
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.word_index, self.maxlen = self.__preprocess()
        self.embedding_matrix = self.__load_embedding()

    def __build_model(self):
        inp = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embedding_matrix),
                      self.embed_size,
                      weights=[self.embedding_matrix],
                      trainable=False)(inp)
        x = Bidirectional(GRU(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
        x = Attention()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self):
        model = self.__build_model()
        checkpoint_dir = os.path.join(self.model_path, 'checkpoints/' + str(int(time.time())) + '/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + '.h5'
        checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        model_trained = model.fit(self.x_train,
                                  self.y_train,
                                  batch_size=256,
                                  epochs=25,
                                  validation_data=[self.x_test, self.y_test],
                                  callbacks=[checkpoint, early])
        plot(model_trained)
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

    def __load_embedding(self):
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(self.word_index) + 1, self.embed_size)
        embeddings[0] = 0

        for word, index in self.word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings
