import itertools
from collections import Counter

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from nlp.preprocess.clean_text import clean_en_text


def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples, positive_labels = __load_data_and_label(positive_data_file)
    negative_examples, negative_labels = __load_data_and_label(negative_data_file, False)

    x_text = positive_examples + negative_examples
    x_text = [clean_en_text(sent).split(' ') for sent in x_text]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def __load_data_and_label(data_file, positive=True):
    examples = list(open(data_file, "r", encoding='utf-8').readlines())
    examples = [s.strip() for s in examples]
    if positive:
        labels = [[0, 1] for _ in examples]
    else:
        labels = [[1, 0] for _ in examples]
    return examples, labels


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
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(positive_data_file, negative_data_file):
    sentences, labels = load_data_and_labels(positive_data_file, negative_data_file)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def preprocess(positive_data_file, negative_data_file):
    x, y, vocabulary, vocabulary_inv = load_data(positive_data_file, negative_data_file)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, vocabulary, vocabulary_inv


class CnnKerasClassifier:

    def __init__(self, embedding_dim=256,
                 filter_sizes=None,
                 num_filters=512,
                 drop=0.5,
                 epochs=100,
                 batch_size=30,
                 positive_data_file=None,
                 negative_data_file=None):
        self.positive_data_file = positive_data_file
        self.negative_data_file = negative_data_file
        self.X_train, self.X_test, self.y_train, self.y_test, self.vocabulary, self.vocabulary_inv = preprocess(
            self.positive_data_file,
            self.negative_data_file)
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        self.sequence_length = self.X_train.shape[1]
        self.vocabulary_size = len(self.vocabulary_inv)
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.drop = drop
        self.epochs = epochs
        self.batch_size = batch_size

    def __build_model(self):
        inputs = Input(shape=(self.sequence_length,), dtype='int32')
        embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim,
                              input_length=self.sequence_length)(inputs)
        reshape = Reshape((self.sequence_length, self.embedding_dim, 1))(embedding)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = Conv2D(self.num_filters,
                          kernel_size=(filter_size, self.embedding_dim),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu')(reshape)
            max_pool = MaxPool2D(pool_size=(self.sequence_length - filter_size + 1, 1), strides=(1, 1),
                                 padding='valid')(
                conv)
            pooled_outputs.append(max_pool)
        concatenated_tensor = Concatenate(axis=1)(pooled_outputs)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(self.drop)(flatten)
        output = Dense(units=2, activation='softmax')(dropout)
        model = Model(inputs=inputs, outputs=output)
        self.checkpoint = ModelCheckpoint('../model/cnn/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='auto')
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        model = self.__build_model()
        model.fit(self.X_train,
                  self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=[self.checkpoint],
                  validation_data=(self.X_test, self.y_test))
