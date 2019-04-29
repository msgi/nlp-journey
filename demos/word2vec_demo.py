from nlp.embedding.word2vec import GensimWord2VecModel

if __name__ == '__main__':
    word_vec_model = GensimWord2VecModel('data/tianlong.txt', 'model/gensim/model.bin')

    print(word_vec_model.similar('段誉'))
    print('**************************************************')
