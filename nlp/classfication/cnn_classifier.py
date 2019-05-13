import itertools
import os
import pickle
from collections import Counter

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Flatten, Dropout, Concatenate
from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from nlp.preprocess.clean_text import clean_en_text
from nlp.utils.plot_model_history import plot


def load_data_and_labels(pos_file, neg_file):
    # 读取文件
    positive_examples = list(open(pos_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # 分词
    x_text = positive_examples + negative_examples
    x_text = [clean_en_text(sent) for sent in x_text]
    x_text = [s.split(' ') for s in x_text]
    # 生成label
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, y


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    stop_words = stopwords.words('english')
    # 构造词典
    word_counts = Counter(itertools.chain(*sentences))
    # 从index到word
    index_word = [word if word not in stop_words else '<PAD/>' for word, num in word_counts.items()]
    index_word = list(sorted(index_word))
    index_word.insert(0, '<PAD/>')
    # 从word到index
    word_index = {x: i for i, x in enumerate(index_word)}
    return word_index


def load_data(pos_file, neg_file):
    # 加载并预处理
    sentences, labels = load_data_and_labels(pos_file, neg_file)
    word_index = build_vocab(sentences)
    sentences = pad_sentences(sentences)
    x = np.array([[word_index[w] if w in word_index else 0 for w in s] for s in sentences])
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test, word_index


class CnnClassifier:
    def __init__(self, model_path,
                 config_file,
                 embedding_dim=256,
                 train=False,
                 pos_file=None,
                 neg_file=None,
                 filter_sizes=None,
                 num_filters=256,
                 drop=0.5,
                 epochs=100,
                 batch_size=128):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5, 6]
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.drop = drop
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.config_file = config_file

        if train:
            # self.x_train, self.y_train, self.x_test, self.y_test, self.word_index = load_data(pos_file, neg_file)
            self.x_train, self.y_train, self.x_test, self.y_test, self.word_index = self.__preprocess()
            self.vocab_size = len(self.word_index)
            self.sequence_length = self.x_train.shape[1]

            self.model = self.train()
        else:
            self.word_index = self.__load_config()
            self.model = self.__load_model()

    def train(self):
        model = self.__build_model()
        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'weights.{epoch:03d}-{val_acc:.4f}.h5'),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')
        early_stop = EarlyStopping(patience=3, verbose=1)
        history = model.fit(self.x_train,
                            self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[checkpoint, early_stop],
                            validation_data=(self.x_test, self.y_test))  # starts training
        plot(history)
        model.save(os.path.join(self.model_path, 'model.h5'))
        self.__save_config()
        return model

    def predict(self, text):
        pass

    def __build_model(self):
        inputs = Input(shape=(self.sequence_length,), dtype='int32')
        embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.sequence_length)(inputs)

        filter_results = []
        for i, filter_size in enumerate(self.filter_sizes):
            c = Conv1D(self.num_filters,
                       kernel_size=filter_size,
                       padding='valid',
                       activation='relu',
                       kernel_regularizer=l2(0.001),
                       name='conv-' + str(i + 1))(embedding)
            max_pool = GlobalMaxPooling1D(name='max-pool-' + str(i + 1))(c)
            filter_results.append(max_pool)
        concat = Concatenate()(filter_results)
        dropout = Dropout(self.drop)(concat)
        output = Dense(units=1,
                       activation='sigmoid',
                       name='dense')(dropout)
        model = Model(inputs=inputs,
                      outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def __load_model(self):
        try:
            model = load_model(os.path.join(self.model_path, 'model.h5'))
        except FileNotFoundError:
            model = None
        return model

    @staticmethod
    def __preprocess():
        from keras.datasets import imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data()

        x_train = pad_sequences(x_train, maxlen=500)
        x_test = pad_sequences(x_test, x_train.shape[1])
        print(type(x_train))
        word_index = imdb.get_word_index()
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')
        return x_train, y_train, x_test, y_test, word_index

    def __load_config(self):
        with open(self.config_file, 'rb') as f:
            word_index = pickle.load(f)
        return word_index

    def __save_config(self):
        with open(self.config_file, 'wb') as f:
            pickle.dump(self.word_index, f)
