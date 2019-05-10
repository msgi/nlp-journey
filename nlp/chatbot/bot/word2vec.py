import os
import gensim
import jieba


def preprocess(train_file):
    # 读取文件内容并分词
    with open(train_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
        sentences = [jieba.lcut(sentence.strip()) for sentence in sentences]

    return sentences


def train(path, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    sentences = preprocess(path + os.sep + 'question.txt')
    model = gensim.models.Word2Vec(sentences, size=128, window=5,
                                   min_count=1, iter=10, workers=4)
    model.save(model_path + os.sep + 'encoder_vector.m')

    print("encoder vector has been generated!")

    sentences = preprocess(path + os.sep + 'answer.txt')
    model = gensim.models.Word2Vec(sentences, size=128, window=5,
                                   min_count=1, iter=10, workers=4)
    model.save(model_path + os.sep + 'decoder_vector.m')

    print(model.wv['喜欢'])
    print("decoder vector has been generated!")
