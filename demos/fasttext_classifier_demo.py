from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = 'data/imdb'
    model = FastTextClassifier('model/fasttext/model')
    model.predict('i don\'t like it because it is too boring')
    model.predict('i like it ! its very interesting')
