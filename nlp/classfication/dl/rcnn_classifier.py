from keras import Input, Model
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Dense

from .basic_classifier import TextClassifier


class TextRCNNClassifier(TextClassifier):
    def build_model(self):
        inputs = Input((self.maxlen,))
        embedding = Embedding(len(self.embeddings),
                              300,
                              weights=[self.embeddings],
                              trainable=False)(inputs)
        x_context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])
        cs = []
        for kernel_size in range(1, 5):
            c = Conv1D(128, kernel_size, activation='relu')(x)
            cs.append(c)
        pools = [GlobalAveragePooling1D()(conv) for conv in cs] + [GlobalMaxPooling1D()(c) for c in cs]
        x = Concatenate()(pools)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, batch_size=512, epochs=20):
        super(TextRCNNClassifier, self).train(128, 2)
