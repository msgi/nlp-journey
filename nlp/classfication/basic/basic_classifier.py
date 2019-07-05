from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import os
from nlp.utils.plot_model_history import plot
import pickle


class TextClassifier:
    """
    基础文本分类器
    """

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 train_path=None):
        self.model_path = model_path
        self.config_path = config_path
        self.word_index = self.load_config()
        if not train:
            self.model = self.load_model()
            if not self.model:
                print('模型找不到：', self.model_path)
        else:
            assert train_path is not None, '训练模式下，train_path不能为None'
            x, y = self.preprocess()

    def build_model(self, input_shape=(784,)):
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def save_model(self, weights_only=True):
        if not self.model:
            if weights_only:
                self.model.save_weights(os.path.join(
                    self.model_path, 'weights.h5'))
            else:
                self.model.save(os.path.join(self.model_path, 'model.h5'))

    def load_model(self, weights_only=True):
        try:
            if weights_only:
                model = self.build_model()
                model.load_weights(os.path.join(self.model_path, 'weights.h5'))
            else:
                model = load_model(os.path.join(self.model_path, 'model.h5'))
        except FileNotFoundError:
            model = None
        return model

    def train(self, x, y,
              test_size=0.2,
              batch_size=512,
              epochs=20):
        model = self.build_model()
        # early_stop配合checkpoint使用，可以得到val_loss最小的模型
        early_stop = self.early_stop()
        checkpoint = self.check_point()
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=test_size, random_state=42)
        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpoint, early_stop],
                            validation_data=(x_val, y_val))
        plot(history)
        return model

    def early_stop(self):
        # 连续三次epoch某个观察项没有改善，则提前退出训练，默认观察项是val_acc
        return EarlyStopping(patience=3, verbose=1)

    def check_point(self):
        return ModelCheckpoint(os.path.join(self.model_path, 'weights.{epoch:03d}-{val_loss:.3f}.h5'),
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True,
                               mode='auto')

    def predict(self, text):
        word_indices = None
        if type(text) == 'str':
            word_indices = [[self.word_index[t] for t in text.split()]]
        elif type(text) == 'list':
            word_indices = [self.word_index[t]
                            for tx in text for t in tx.split()]

        if not word_indices:
            return self.model.predict(word_indices)
        else:
            return []

    def load_config(self):
        with open(self.config_path, 'rb') as f:
            word_index = pickle.load(f)
        return word_index

    def save_config(self):
        with open(self.config_path, 'wb') as f:
            pickle.dump(self.word_index, f)

    def summary(self):
        self.build_model().summary()
