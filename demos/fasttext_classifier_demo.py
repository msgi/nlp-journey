from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = '/home/msg/workspace/pycharm/nlp-tutorials/demos/data/sentiment'
    model = FastTextClassifier('./model/fasttext/classifier', train=True, file_path=train_path)
    model.predict('我 怎么 借不了 款')
