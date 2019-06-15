from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = 'data/imdb/'
    model = FastTextClassifier('./model/fasttext/classifier', train=True, file_path=train_path)
    model.predict('')
