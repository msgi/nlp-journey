import codecs
import os

import gensim
import jieba
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Embedding
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences

from nlp.chatbot.seq2seq import AttentionSeq2Seq
from .data_preprocess import __PAD__
from .data_preprocess import __UNK__
from .data_preprocess import __VOCAB__
from .data_preprocess import preprocess


class ChatBot:

    def __init__(self, train_file, model_path, decoder_vector_path,encoder_vector_path):
        self.train_file = train_file
        self.questions, self.answers = self.__read_file()

        self.enc_vocab_size = 20000
        self.dec_vocab_size = 20000
        self.enc_input_length = 50
        self.dec_output_length = 50
        self.enc_embedding_length = 128
        self.dec_embedding_length = 128
        self.hidden_dim = 100
        self.layer_shape = (2, 1)
        self.epsilon = 1e-6
        self.batch_size = 128
        self.epochs = 500
        self.decoder_vector_path = decoder_vector_path
        self.encoder_vector_path=encoder_vector_path

        self.decoder_word2vec_model = self.__load_word2vec_model(decoder_vector_path)
        self.encoder_vec_model = self.__load_word2vec_model(self.encoder_vector_path)

        self.enc_sequences, self.enc_word_index, self.enc_index_word = preprocess(self.questions,
                                                                                  self.enc_input_length,
                                                                                  self.enc_vocab_size)
        self.dec_sequences, self.dec_word_index, self.dec_index_word = preprocess(self.answers,
                                                                                  self.dec_output_length,
                                                                                  self.dec_vocab_size,
                                                                                  post=True)
        self.vocab_size = len(self.enc_word_index)
        self.model_path = model_path
        self.model = self.load_model()
        if not self.model:
            self.model = self.train()

    def __read_file(self):
        """
        处理训练数据,训练数据格式:question###answer
        :return: questions, answers
        """
        questions = []
        answers = []
        with codecs.open(self.train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for l in lines:
            split = l.split('###')
            questions.append(split[0].strip())
            answers.append(split[1].strip())
        return questions, answers

    # 获取索引表示的输入和embedding表示的输出
    def generate_batch(self, batch_size=None):
        dec_useful_words = list(self.decoder_word2vec_model.wv.vocab.keys())
        batch_count = 0
        x_train = []
        y = []
        num = 0
        while True:
            source_list = self.enc_sequences[num]
            target_str_list = self.dec_sequences[num]
            target_list = []
            for data in target_str_list:
                word = self.dec_index_word[data]
                if word in dec_useful_words:
                    word_embedding = self.decoder_word2vec_model.wv[word]
                elif word == __VOCAB__[0]:
                    word_embedding = np.zeros(self.dec_embedding_length)
                else:
                    word_embedding = np.array([1.0] * self.dec_embedding_length)
                # 归一化
                std_number = np.std(word_embedding)
                if (std_number - self.epsilon) < 0:
                    word_embedding_scale = np.zeros(self.dec_embedding_length)
                else:
                    word_embedding_scale = (word_embedding - np.mean(word_embedding)) / std_number
                target_list.append(word_embedding_scale)
            x_train.append(source_list)
            y.append(target_list)
            batch_count += 1
            if batch_count == batch_size:
                batch_count = 0
                x_array = np.array(x_train)
                y_array = np.array(y)
                yield (x_array, y_array)
                x_train = []
                y = []
            num += 1
            if num == 63:
                num = 0

    @staticmethod
    def __load_word2vec_model(word2vec_path):
        return gensim.models.Word2Vec.load(word2vec_path)

    def train(self):
        model = self.__build_model()
        documents_length = len(self.enc_sequences)
        if self.batch_size > documents_length:
            print("ERROR--->" + u"语料数据量过少，请再添加一些")
            return None
        model.fit_generator(
            generator=self.generate_batch(batch_size=self.batch_size),
            steps_per_epoch=int(documents_length / self.batch_size),
            epochs=self.epochs,
            verbose=1)
        model.save_weights(self.model_path)
        return model

    def __build_model(self, training=True):
        # 初始化词向量
        embedding_matrix = self.get_encoder_embedding()
        embedding_layer = Embedding(
            self.vocab_size,
            self.enc_embedding_length,
            weights=[embedding_matrix],
            input_length=self.enc_input_length,
            trainable=training,
            name='encoder_embedding')

        enc_normalization = BatchNormalization(epsilon=self.epsilon)

        seq2seq = AttentionSeq2Seq(
            bidirectional=False,
            output_dim=self.dec_embedding_length,
            hidden_dim=self.hidden_dim,
            output_length=self.dec_output_length,
            input_shape=(self.enc_input_length, self.enc_embedding_length),
            depth=self.layer_shape)

        model = Sequential()
        model.add(embedding_layer)
        model.add(enc_normalization)
        model.add(seq2seq)

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def get_encoder_embedding(self):
        embedding_list = []
        for key, value in self.enc_index_word.items():
            if key == __PAD__:
                embedding_list.append(np.array([0.0] * self.enc_embedding_length))
            elif key == __UNK__:
                embedding_list.append(np.array([1.0] * self.enc_embedding_length))
            elif value in self.encoder_vec_model.wv.vocab:
                embedding_list.append(self.encoder_vec_model.wv[value])
            else:
                embedding_list.append(np.array([1.0] * self.enc_embedding_length))
        return np.array(embedding_list)

    def predict(self, questions):
        question_ids = self.__to_data_ids(questions)
        return self.__predict_text(question_ids)

    def __predict_text(self, enc_embedding):
        """
        输出答案
        :param enc_embedding: 转换为id的问题
        :return: 字符列表
        """

        dec_useful_words = tuple(self.decoder_word2vec_model.wv.vocab.keys())
        prediction = self.model.predict_on_batch(enc_embedding)

        prediction_words_list = []
        for elem in prediction:
            prediction_words = []
            for vec in elem:
                dec_dis_list = []
                mse = self.calculate_mse(vec, np.zeros(self.dec_embedding_length))
                dec_dis_list.append(mse)
                for dec_word in dec_useful_words:
                    mse = self.calculate_mse(vec, self.decoder_word2vec_model.wv[dec_word])
                    dec_dis_list.append(mse)
                index = np.argmin(dec_dis_list)
                if index == 0:
                    word = __VOCAB__[0]
                else:
                    word = dec_useful_words[int(index - 1)]
                prediction_words.append(word)
            prediction_words_list.append(prediction_words)

        return prediction_words_list

    def calculate_mse(self, src_vec, des_vec):
        """
        计算两个向量的均方差
        :param src_vec:
        :param des_vec:
        :return:
        """
        std_number = np.std(des_vec)
        if (std_number - self.epsilon) < 0:
            norm_des_vec = np.zeros(self.dec_embedding_length)
        else:
            norm_des_vec = (des_vec - np.mean(des_vec)) / std_number
        err = np.square(src_vec - norm_des_vec)
        mse = np.sum(err)
        return mse

    def __to_data_ids(self, text_list):
        """
        将问题转为字典id的形式
        :param text_list:
        :return:
        """
        enc_padding_ids_list = []
        for text in text_list:
            words_list = list(jieba.cut(text.strip()))
            enc_ids = [self.enc_word_index.get(word, __UNK__) for word in words_list]
            if len(enc_ids) > self.enc_input_length:
                enc_ids = enc_ids[:self.enc_input_length]
            enc_padding_ids = pad_sequences(enc_ids, maxlen=self.enc_input_length, value=__PAD__)
            enc_padding_ids_list.append(np.array(enc_padding_ids))
        return np.array(enc_padding_ids_list)

    def load_model(self):
        if os.path.exists(self.model_path):
            model = self.__build_model(training=False)
            model.load_weights(self.model_path)
        else:
            model = None
        return model
