from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, TimeDistributed, Dense, np
from keras_preprocessing.sequence import pad_sequences

from nlp.layers.attention import Attention


class HANClassifier:
    def __init__(self, maxlen_sentence=16,
                 maxlen_word=25,
                 max_features=5000,
                 embedding_dims=50,
                 class_num=1,
                 batch_size=32,
                 epochs=10,
                 last_activation='sigmoid'):
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.batch_size = batch_size
        self.epochs = epochs

    def __build_model(self):
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen_word)(input_word)
        x_word = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_word)
        x_word = Attention()(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(shape=(self.maxlen_sentence, self.maxlen_word))
        x_sentence = TimeDistributed(model_word)(input)
        x_sentence = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_sentence)
        x_sentence = Attention()(x_sentence)

        output = Dense(self.class_num, activation=self.last_activation)(x_sentence)
        model = Model(inputs=input, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        model = self.__build_model()
        early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=[early_stopping],
                  validation_data=(x_test, y_test))

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
