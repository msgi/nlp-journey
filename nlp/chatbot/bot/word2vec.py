import os
import gensim
import jieba


def preprocess(train_file):
    # 读取文件内容并分词
    encoder_lines = []
    decoder_lines = []
    with open(train_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
        for s in sentences:
            ss = s.split("###")
            encoder_lines.append(ss[0])
            decoder_lines.append(ss[1])
        encoder_lines = [jieba.lcut(sentence.strip()) for sentence in encoder_lines]
        decoder_lines = [jieba.lcut(sentence.strip()) for sentence in decoder_lines]

    return encoder_lines, decoder_lines


def train(path, model_path):
    encoder_lines, decoder_lines = preprocess(path + os.sep + 'train.txt')
    model = gensim.models.Word2Vec(encoder_lines, size=128, window=5,
                                   min_count=1, iter=10, workers=4)
    model.save(model_path + os.sep + 'encoder_vector.m')

    print("encoder vector has been generated!")

    model = gensim.models.Word2Vec(decoder_lines, size=128, window=5,
                                   min_count=1, iter=10, workers=4)
    model.save(model_path + os.sep + 'decoder_vector.m')

    print(model.wv['喜欢'])
    print("decoder vector has been generated!")
