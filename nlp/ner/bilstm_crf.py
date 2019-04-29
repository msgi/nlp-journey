import pickle
import platform
from collections import Counter
import os

import numpy as np
from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras_contrib import losses, metrics
from keras_contrib.layers import CRF
import logging
from nlp.utils.basic_log import Log

log = Log(logging.INFO)


def _process_data(data, word2idx, chunk_tags, max_len=None, one_hot=False):
    if max_len is None:
        max_len = max(len(s) for s in data)
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, max_len)
    y_chunk = pad_sequences(y_chunk, max_len, value=-1)

    if one_hot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def _parse_data(file_path):
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'
    with open(file_path, 'rb') as f:
        string = f.read().decode('utf-8')
        data = [[row.split() for row in sample.split(split_text)] for sample in
                string.strip().split(split_text + split_text)]
    return data


class BiLSTMNamedEntityRecognition:
    def __init__(self,
                 model_path,
                 config_path,
                 embed_dim=200,
                 rnn_units=200,
                 epochs=1,
                 batch_size=64,
                 train=False,
                 file_path=None):
        self.model_path = model_path
        self.config_path = config_path
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        # 词性tag
        self.tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

        # 非训练模式，直接加载模型
        if not train:
            self.word2idx = self.__load_config()
            self.model = self.__load_model()
            assert self.model is not None, '训练模型无法获取'
        else:
            self.file_path = file_path
            self.batch_size = batch_size
            (self.train_x, self.train_y), (self.test_x, self.test_y), self.word2idx = self.__load_data()
            self.epochs = epochs
            self.model = self.train()

    # 训练
    def train(self):
        model = self.__build_model()
        model.fit(self.train_x,
                  self.train_y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=[self.test_x, self.test_y])
        model.save_weights(self.model_path)
        self.__save_config()
        return model

    # 加载训练好的模型
    def __load_model(self):
        try:
            model = self.__build_model()
            model.load_weights(self.model_path)
        except FileNotFoundError:
            log.error('没有找到模型文件')
            model = None
        return model

    # 加载词表
    def __load_config(self):
        with open(self.config_path, 'rb') as config:
            word2idx = pickle.load(config)
        return word2idx

    # 保存词表
    def __save_config(self):
        with open(self.config_path, 'wb') as out:
            pickle.dump(self.word2idx, out)

    # 识别句子中的实体
    def predict(self, predict_text):
        # predict_text = ''
        sent, length = self.__preprocess_data(predict_text)
        raw = self.model.predict(sent)[0][-length:]
        result = np.argmax(raw, axis=1)
        result_tags = [self.tags[i] for i in result]

        per, loc, org = '', '', ''
        for s, t in zip(predict_text, result_tags):
            if t in ('B-PER', 'I-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s
        results = ['person:' + per, 'location:' + loc, 'organization:' + org]
        print(results)
        return results

    # 构造模型
    def __build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.word2idx), self.embed_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(self.rnn_units // 2, return_sequences=True)))
        crf = CRF(len(self.tags), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=losses.crf_loss, metrics=[metrics.crf_accuracy])
        return model

    # 训练数据预处理
    def __load_data(self):
        train = _parse_data(os.path.join(self.file_path, 'train.data'))
        test = _parse_data(os.path.join(self.file_path, 'test.data'))

        # 统计每个字出现的频次
        word_counts = Counter(row[0].lower() for sample in train for row in sample)
        vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
        word2idx = dict((w, i) for i, w in enumerate(vocab))

        train = _process_data(train, word2idx, self.tags)
        test = _process_data(test, word2idx, self.tags)
        return train, test, word2idx

    # 预测的时候，进行数据处理转换
    def __preprocess_data(self, data, max_len=100):
        x = [self.word2idx.get(w[0].lower(), 1) for w in data]
        length = len(x)
        x = pad_sequences([x], max_len)
        return x, length
