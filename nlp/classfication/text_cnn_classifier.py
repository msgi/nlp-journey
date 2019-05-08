from keras import Input, Model
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense


class TextCNNClassifier:

    def __init__(self, embedding_matrix,
                 embeddings_dim):
        self.embedding_matrix = embedding_matrix
        self.embeddings_dim = embeddings_dim
        pass

    def __build_model(self):
        embedding_layer = Embedding(len(self.embedding_matrix),
                                    int(self.embeddings_dim),
                                    weights=[self.embedding_matrix],
                                    trainable=False)

        print('Training model.')
        sequence_input = Input(shape=(MAX_NEWS_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dropout(0.4)(embedded_sequences)
        x = Conv1D(25, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Flatten()(x)
        x = Dense(25, activation='relu')(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        model.summary()
        return model

    def train(self):
        pass


