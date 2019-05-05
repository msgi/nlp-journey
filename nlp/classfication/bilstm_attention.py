import logging

from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

from nlp.layers.attention import Attention
from keras.datasets import imdb
import numpy as np
from nlp.utils.basic_log import Log

log = Log(logging.INFO)


class BiLSTMAttentionModel:

    def __init__(self, embedding_file,
                 embed_size=300,
                 max_features=10000):
        self.max_features = max_features
        self.embed_size = embed_size
        self.embedding_file = embedding_file
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.word_index, self.maxlen = self.__preprocess()
        self.embedding_matrix = self.__load_embedding()

    def __build_model(self):
        inp = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embedding_matrix),
                      self.embed_size,
                      weights=[self.embedding_matrix],
                      trainable=False)(inp)
        x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                               recurrent_dropout=0.25))(x)
        x = Attention(self.maxlen)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self):
        model = self.__build_model()
        file_path = ".model.hdf5"
        ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        model.fit(self.x_train,
                  self.y_train,
                  batch_size=256,
                  epochs=15,
                  validation_data=[self.x_test, self.y_test],
                  callbacks=[ckpt, early])

    @staticmethod
    def __preprocess():
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        x_train = train_data
        x_test = test_data
        word_index = imdb.get_word_index()

        maxlen1 = max([len(x) for x in x_train])
        maxlen2 = max([len(x) for x in x_test])

        print(maxlen1, maxlen2)

        maxlen = max(maxlen1, maxlen2)
        y_train = np.asarray(train_labels).astype('float32')
        y_test = np.asarray(test_labels).astype('float32')

        return (x_train, y_train), (x_test, y_test), word_index, maxlen

    def __load_embedding(self):
        log.info('加载词向量...')
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(self.word_index) + 1, self.embed_size)
        embeddings[0] = 0

        for word, index in self.word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings
