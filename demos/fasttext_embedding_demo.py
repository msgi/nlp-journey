from nlp.embedding.fasttext_model import FastTextModel

if __name__ == '__main__':
    # cbow 模型
    # model = FastTextModel('data/tianlong_seg.txt', 'model/fasttext/model', model_type='cbow')
    # skipgram 模型
    model = FastTextModel('data/tianlong_seg.txt', 'model/fasttext/model')
    print(model.vector('段誉'))
