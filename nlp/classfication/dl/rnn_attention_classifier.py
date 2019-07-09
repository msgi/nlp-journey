from keras import Model

from nlp.layers.attention import Attention
from .basic_classifier import TextClassifier
from keras.layers import Dropout, Dense, Embedding, Input, Bidirectional, LSTM


class TextRNNAttentionClassifier(TextClassifier):

    def build_model(self):
        inputs = Input(shape=(self.maxlen,))
        output = Embedding(len(self.embeddings),
                           300,
                           weights=[self.embeddings],
                           trainable=False)(inputs)
        output = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(output)
        output = Attention()(output)
        output = Dense(128, activation="relu")(output)
        output = Dropout(0.25)(output)
        output = Dense(1, activation="sigmoid")(output)
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self, batch_size=512, epochs=20):
        super(TextRNNAttentionClassifier,self).train(128,3)
