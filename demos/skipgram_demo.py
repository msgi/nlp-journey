from nlp.embedding.skipgram import FastTextSkipGramModel, process_data

if __name__ == '__main__':
    process_data('data/tianlong.txt','data/tianlong_seg.txt')
    model = FastTextSkipGramModel('data/tianlong_seg.txt', 'model/fasttext/model')
    print(model.vector('段誉'))
