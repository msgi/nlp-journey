from nlp.classfication.svm_classifier import SVMClassifier

if __name__ == '__main__':
    svm_model = SVMClassifier('model/svm/model.pkl', 'data/imdb/aclImdb.txt', train=True)
    # svm_model = SVMClassifier('model/svm/model.pkl')
    svm_model.predict(['i like it ! its very interesting', 'I don\'t like it, it\'s boring'])
