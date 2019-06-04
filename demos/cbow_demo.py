from nlp.embedding.cbow import FastTextCBowModel
from nlp.utils.pre_process import seg_to_file

if __name__ == '__main__':
    seg_to_file('data/tianlong.txt', 'data/tianlong_seg.txt')
    model = FastTextCBowModel('data/tianlong_seg.txt', 'model/fasttext/model')
    print(model.vector('段誉'))
