from keras import Model
from keras.layers import Dropout, Dense, Embedding, BatchNormalization, CuDNNLSTM, Input, Bidirectional

from .basic_classifier import TextClassifier


class TextRnnClassifier(TextClassifier):
    def __init__(self, model_path, config_path, train, vector_path):
        super(TextRnnClassifier, self).__init__(model_path, config_path, train, vector_path)

    def build_model(self):
        inputs = Input(shape=(self.maxlen,))
        x = Embedding(len(self.embeddings),
                      300,
                      weights=[self.embeddings],
                      trainable=False)(inputs)
        x = Bidirectional(CuDNNLSTM(150))(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        y = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
