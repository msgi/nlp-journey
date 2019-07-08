from keras import Input, Model
from keras.layers import Concatenate, Dropout, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.regularizers import l2

from .basic_classifier import TextClassifier


class TextCnnClassifier(TextClassifier):

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 vector_path=None,
                 filter_sizes=None,
                 num_filters=256,
                 drop=0.5):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5, 6]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.drop = drop
        super(TextCnnClassifier, self).__init__(model_path, config_path, train, vector_path)

    def build_model(self, input_shape=(500,)):
        inputs = Input(shape=input_shape, dtype='int32')
        embedding = Embedding(len(self.embeddings),
                              300,
                              weights=[self.embeddings],
                              trainable=False)(inputs)

        filter_results = []
        for i, filter_size in enumerate(self.filter_sizes):
            c = Conv1D(self.num_filters,
                       kernel_size=filter_size,
                       padding='valid',
                       activation='relu',
                       kernel_regularizer=l2(0.001),
                       name='conv-' + str(i + 1))(embedding)
            max_pool = GlobalMaxPooling1D(name='max-pool-' + str(i + 1))(c)
            filter_results.append(max_pool)
        concat = Concatenate()(filter_results)
        dropout = Dropout(self.drop)(concat)
        output = Dense(units=1,
                       activation='sigmoid',
                       name='dense')(dropout)
        model = Model(inputs=inputs,
                      outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
