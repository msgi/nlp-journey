# coding:utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import pandas as pd
from nlp.utils.clean_text import clean_en_text
from nlp.utils.basic_log import Log
import logging

log = Log(logging.INFO)


class SVMClassifier(object):
    """
    这个类,是用svm对文本进行分类.
    1. 用TF-IDF计算权重值
    2. 用卡方检验获取特征
    3. 用SVM进行分类训练
    """

    def __init__(self, model_file,
                 train_path=None,
                 train=False):
        """
        初始化参数
        :param train_path: 训练路径：数据以“x##y”的格式分隔，x为分好词的数据，y为数据标签
        :param model_file: 模型保存路径
        """
        self.model_file = model_file
        # 先读取训练好的model,如果读取不到,则重新训练
        if not train:
            self.tf_idf_model, self.chi_model, self.clf_model = self.read_model()
            assert self.tf_idf_model is not None, '训练的模型不存在,请确认后再试'
        else:
            assert train_path is not None, '训练模式下, 训练数据不能为None'
            self.train_path = train_path
            self.tf_idf_model, self.chi_model, self.clf_model = self.train_model()
            self.save_model()

    def predict(self, texts):
        """
        根据模型预测某文件的分类
        :param texts: 要分类的文本
        :return: 返回分类
        """
        texts = [clean_en_text(t) for t in texts]
        tf_vector = self.tf_idf_model.transform(texts)
        chi_vector = self.chi_model.transform(tf_vector)
        out = self.clf_model.predict(chi_vector)
        print('推理结果：', out)
        return out

    def read_model(self):
        try:
            with open(self.model_file, 'rb') as f:
                (tf_idf_model, chi_model, clf_model) = pickle.load(f)
        except FileNotFoundError:
            log.error('没有找到模型文件')
            tf_idf_model = None
            chi_model = None
            clf_model = None
        return tf_idf_model, chi_model, clf_model

    def train_model(self, test_size=0.2):
        """
        训练模型,简单地将生成的TF-IDF数据,chi提取后的特征,以及svm算法模型写入到了磁盘中
        :return: 返回训练好的模型
        """
        data_set = pd.read_csv(self.train_path,
                               sep='##',
                               encoding='utf-8',
                               header=None,
                               engine='python')
        data_set = data_set.dropna()
        chi_features, tf_idf_model, chi_model = self.__select_features(data_set)
        x_train, x_test, y_train, y_test = train_test_split(chi_features,
                                                            data_set[1],
                                                            test_size=test_size,
                                                            random_state=42)
        # 这里采用的是线性分类模型,如果采用rbf径向基模型,速度会非常慢.
        clf_model = svm.SVC(kernel='linear', verbose=True)
        print(clf_model)
        clf_model.fit(x_train, y_train)
        score = clf_model.score(x_test, y_test)
        print('测试准确率:', score)
        return tf_idf_model, chi_model, clf_model

    @staticmethod
    def __select_features(data_set):
        dataset = [clean_en_text(data) for data in data_set[0]]
        tf_idf_model = TfidfVectorizer(ngram_range=(1, 1),
                                       binary=True, 
                                       sublinear_tf=True)
        tf_vectors = tf_idf_model.fit_transform(dataset)

        # 选出前1/5的词用来做特征
        k = int(tf_vectors.shape[1] / 6)
        chi_model = SelectKBest(chi2, k=k)
        chi_features = chi_model.fit_transform(tf_vectors, data_set[1])
        print('tf-idf:\t\t' + str(tf_vectors.shape[1]))
        print('chi:\t\t' + str(chi_features.shape[1]))

        return chi_features, tf_idf_model, chi_model

    def save_model(self):
        with open(self.model_file, "wb") as file:
            pickle.dump((self.tf_idf_model, self.chi_model, self.clf_model), file)
        if file:
            file.close()
