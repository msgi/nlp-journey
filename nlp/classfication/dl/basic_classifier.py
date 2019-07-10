from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.datasets import imdb
from keras.layers import Input, Dense, Embedding, Flatten, Lambda
from keras.models import Model
import os

from keras.utils import to_categorical

from nlp.utils.plot_model_history import plot
import pickle
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
import keras.backend as K
from sklearn.model_selection import train_test_split
from collections import Counter


class TextClassifier:
    """
    基础文本分类器
    """

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 train_file_path=None,
                 vector_path=None):
        self.model_path = model_path
        self.config_path = config_path
        if not train:
            self.word_index, self.maxlen, self.embeddings = self.load_config()
            self.model = self.load_model()
            if not self.model:
                print('模型找不到：', self.model_path)
        else:
            self.vector_path = vector_path
            self.train_file_path = train_file_path
            self.x_train, self.y_train, self.x_test, self.y_test, self.word_index= self.load_data()
            self.maxlen = self.x_train.shape[1]
            _, _, self.embeddings = self.load_config()
            if len(self.embeddings) == 0:
                self.embeddings = self.load_vector()
                self.save_config()
            self.model = self.train()
            self.save_model()

    # 全连接的一个简单的网络,仅用来作为基类测试,效果特别差
    def build_model(self):
        inputs = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embeddings),
                      300,
                      weights=[self.embeddings],
                      trainable=False)(inputs)
        x = Lambda(lambda t: K.mean(t, axis=1))(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def save_model(self, weights_only=True):
        if self.model:
            if weights_only:
                self.model.save_weights(os.path.join(self.model_path, 'weights.h5'))
            else:
                self.model.save(os.path.join(self.model_path, 'model.h5'))

    def load_model(self, weights_only=True):
        try:
            if weights_only:
                model = self.build_model()
                model.load_weights(os.path.join(self.model_path, 'weights.h5'))
            else:
                model = load_model(os.path.join(self.model_path, 'model.h5'))
        except FileNotFoundError:
            model = None
        return model

    def train(self, batch_size=512, epochs=20):
        model = self.build_model()
        # early_stop配合checkpoint使用，可以得到val_loss最小的模型
        early_stop = EarlyStopping(patience=3, verbose=1)
        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'weights.{epoch:03d}-{val_loss:.3f}.h5'),
                                     verbose=1,
                                     save_best_only=True)
        history = model.fit(self.x_train,
                            self.y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpoint, early_stop],
                            validation_data=(self.x_test, self.y_test))
        plot(history)
        return model

    def predict(self, text):
        indices = None
        if isinstance(text, str):
            indices = [[self.word_index[t] if t in self.word_index.keys() else 0 for t in text.split()]]
        elif isinstance(text, list):
            indices = [[self.word_index[t] if t in self.word_index.keys() else 0 for t in tx.split()] for tx in text]
        if indices:
            indices = pad_sequences(indices, 500)
            return self.model.predict(indices)
        else:
            return []

    def load_config(self):
        try:
            with open(self.config_path, 'rb') as f:
                (word_index, maxlen, embeddings) = pickle.load(f)
        except FileNotFoundError:
            word_index, maxlen, embeddings = None, None, np.array([])
        return word_index, maxlen, embeddings

    def save_config(self):
        with open(self.config_path, 'wb') as f:
            pickle.dump((self.word_index, self.maxlen, self.embeddings), f)

    def summary(self):
        self.build_model().summary()

    def load_vector(self):
        return self.load_vector_en()

    def load_vector_zh(self):
        word2vec = KeyedVectors.load_word2vec_format(self.vector_path, binary=False)
        return self.load_vectors(word2vec)

    def load_vector_en(self):
        word2vec = KeyedVectors.load_word2vec_format(self.vector_path, binary=True)
        return self.load_vectors(word2vec)

    def load_vectors(self, word2vec):
        embeddings = 1 * np.random.randn(len(self.word_index) + 1, 300)
        embeddings[0] = 0
        for word, index in self.word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    def load_stopwords(self):
        from nltk.corpus import stopwords
        return stopwords.words('english')

    # 默认选用keras自带的处理好的数据来做模拟分类
    def load_data(self):
        return self.load_data_from_keras()

    def load_data_from_keras(self, maxlen=500):
        (x_train, y_train), (x_test, y_test) = imdb.load_data()

        word_index = imdb.get_word_index()

        x_train = pad_sequences(x_train, maxlen=maxlen)
        x_test = pad_sequences(x_test, x_train.shape[1])
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')
        return x_train, y_train, x_test, y_test, word_index

    # 用自己的数据集做训练（格式：分好词的句子##标签，如：我 很 喜欢 这部 电影#pos）
    def load_data_from_scratch(self, test_size=0.2, maxlen=100):
        stopwords = self.load_stopwords()
        with open(self.train_file_path, 'r', encoding='utf-8') as file:
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

        max_sentence_length = max([len(words) for words in x])
        maxlen = maxlen if max_sentence_length > maxlen else max_sentence_length

        x_data = [[word_index[word] for word in words if word in word_index.keys() and word not in stopwords] for words
                  in x]
        x_data = pad_sequences(x_data, maxlen=maxlen)

        y_data = to_categorical(y)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
        return x_train, y_train, x_test, y_test, word_index
