from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, Concatenate, Conv1D, GlobalAveragePooling1D, Dense, \
    GlobalMaxPooling1D


class RCNNClassifier:
    def __init__(self, maxlen,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 batch_size=128,
                 epochs=10,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        model = self.__build_model()
        early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[early_stopping],
                  validation_data=(x_test, y_test))
        pass

    def __build_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)

        x_context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])

        cs = []
        for kernel_size in range(1, 5):
            c = Conv1D(128, kernel_size, activation='relu')(x)
            cs.append(c)
        pools = [GlobalAveragePooling1D()(conv) for conv in cs] + [GlobalMaxPooling1D()(c) for c in cs]
        x = Concatenate()(pools)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model
