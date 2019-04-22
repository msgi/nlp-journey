from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = '../data/imdb'
    model1 = FastTextClassifier('../model/fasttext/model')
    model1.predict('i don\'t like it because it is too boring')
    model1.predict('i like it ! its very interesting')
