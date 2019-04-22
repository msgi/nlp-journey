import pickle
import platform
from collections import Counter

import numpy as np
from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF


def load_data():
    train = _parse_data(open('data/ner/train.data', 'rb'))
    test = _parse_data(open('data/ner/test.data', 'rb'))

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # save initial config data
    with open('../model/config.pkl', 'wb') as out_p:
        pickle.dump((vocab, chunk_tags), out_p)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _process_data(data, vocab, chunk_tags, max_len=None, one_hot=False):
    if max_len is None:
        max_len = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    print(y_chunk[:20])

    x = pad_sequences(x, max_len)  # left padding

    y_chunk = pad_sequences(y_chunk, max_len, value=-1)

    if one_hot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def _parse_data(fh):
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def process_data(data, vocab, max_len=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], max_len)  # left padding
    return x, length


class NER:

    def __init__(self):
        self.embed_dim = 200
        self.biRNN_units = 200
        (self.train_x, self.train_y), (self.test_x, self.test_y), (self.vocab, self.chunk_tags) = load_data()
        self.epochs = 10

    def train(self):
        model = self.__build_model()
        model.fit(self.train_x,
                  self.train_y,
                  batch_size=16,
                  epochs=self.epochs,
                  validation_data=[self.test_x, self.test_y])
        model.save('../model/crf.h5')

    def val(self):
        model, (vocab, chunk_tags) = self.__build_model()
        predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
        sent, length = process_data(predict_text, vocab)
        model.load_weights('../model/crf.h5')
        raw = model.predict(sent)[0][-length:]
        result = [np.argmax(row) for row in raw]
        result_tags = [chunk_tags[i] for i in result]

        per, loc, org = '', '', ''

        for s, t in zip(predict_text, result_tags):
            if t in ('B-PER', 'I-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s

        print(['person:' + per, 'location:' + loc, 'organzation:' + org])

    def __build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocab), self.embed_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(self.biRNN_units // 2, return_sequences=True)))
        crf = CRF(len(self.chunk_tags), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model
