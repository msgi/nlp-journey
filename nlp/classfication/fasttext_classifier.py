# coding:utf-8

import fasttext
from nlp.utils.clean_text import clean_zh_text, clean_en_text
import os
from nlp.utils.basic_log import Log
import logging


log = Log(logging.INFO)


class FastTextClassifier:
    """
    利用fasttext来对文本进行分类
    """

    def __init__(self, model_path, train=False, file_path=None):
        """
        初始化
        :param file_path: 训练数据路径
        :param model_path: 模型保存路径
        """
        self.model_path = model_path
        if not train:
            self.model = self.load(self.model_path)
            assert self.model is not None, '训练模型无法获取'
        else:
            assert file_path is not None, '训练时, file_path不能为None'
            self.train_path = os.path.join(file_path, 'train.txt')
            self.test_path = os.path.join(file_path, 'test.txt')
            self.model = self.train()

    def train(self):
        """
        训练:参数可以针对性修改,进行调优,目前采用的参数都是默认参数,可能不适合具体领域场景
        """
        model = fasttext.supervised(self.train_path, self.model_path, label_prefix="__label__", lr=0.01)

        test_result = model.test(self.test_path)
        print('准确率: ', test_result.precision)
        return model

    def predict(self, text):
        """
        预测一条数据,由于fasttext获取的参数是列表,如果只是简单输入字符串,会将字符串按空格拆分组成列表进行推理
        :param text: 待分类的数据
        :return: 分类后的结果
        """
        output = self.model.predict([text])
        print('predict:', output)
        return output

    def load(self, model_path):
        """
        加载训练好的模型
        :param model_path: 训练好的模型路径
        :return:
        """
        if os.path.exists(self.model_path + '.bin'):
            return fasttext.load_model(model_path + '.bin', label_prefix='__label__')
        else:
            return None


def clean(file_path):
    """
    清理文本, 然后利用清理后的文本进行训练
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_clean = []
        for line in lines:
            line_list = line.split('__label__')
            lines_clean.append(clean_en_text(line_list[0]) + ' __label__' + line_list[1])

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_clean)
