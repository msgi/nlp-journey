from keras import backend
from keras.layers import Conv1D, Dense, Input, Lambda, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.models import Model


class BiLSTMCnnClassifier:

    def __init__(self, max_tokens,
                 embedding_dim,
                 hidden_dim_1=200,
                 hidden_dim_2=100,
                 num_classes=10):
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_classes = num_classes
        self.embeddings = None
        pass

    def __load_model(self):
        pass

    def __build_model(self):
        document = Input(shape=(None,), dtype="int32")
        left_context = Input(shape=(None,), dtype="int32")
        right_context = Input(shape=(None,), dtype="int32")

        embedding = Embedding(self.max_tokens + 1, self.embedding_dim, weights=[self.embeddings], trainable=False)
        doc_embedding = embedding(document)
        l_embedding = embedding(left_context)
        r_embedding = embedding(right_context)

        forward = LSTM(self.hidden_dim_1, return_sequences=True)(l_embedding)
        backward = LSTM(self.hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)
        backward = Lambda(lambda x: backend.reverse(x, axes=1))(backward)
        together = concatenate([forward, doc_embedding, backward], axis=2)

        semantic = Conv1D(self.hidden_dim_2, kernel_size=1, activation="tanh")(together)

        pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(self.hidden_dim_2,))(semantic)

        output = Dense(self.num_classes, input_dim=self.hidden_dim_2, activation="softmax")(pool_rnn)

        model = Model(inputs=[document, left_context, right_context], outputs=output)
        model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        pass

    def train(self):
        pass

    def predict(self):
        pass
