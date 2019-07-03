from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = 'data/imdb/'
    model = FastTextClassifier('./model/fasttext/classifier', train=True, file_path=train_path)
    model.predict('this is the weepy that beaches never was as much as i wanted to love beaches it always seemed too hurried for me to feel for it its soundtrack is one of my favorite albums though stella on the other hand moves at a slower and occasionally too slow pace and though it s somewhat manipulative in its tears inducing tale about a self sacrificial mother it works because bette and the rest of the cast turn in great performances')
