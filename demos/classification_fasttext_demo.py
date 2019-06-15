from nlp.classfication.fasttext_classifier import FastTextClassifier

if __name__ == '__main__':
    train_path = 'data/imdb/'
    model = FastTextClassifier('./model/fasttext/classifier', train=True, file_path=train_path)
    model.predict('unlike another user who said this movie sucked and that olivia hussey was terrible i disagree br br this movie was amazing olivia hussey is awesome in everything she s in yeah she may be older now because many remember her from romeo and juliet but she s wonderful br br this story line may be used quite often but it s a unique movie and i ll fight back on anyone who disagrees i enjoyed this movie just as much as i have any other olivia hussey movie olivia s my girl and i love her work br br i saw this for the first time on saturday 4 14 07 and fell in love with it not only because s it s an olivia movie but because of it s unique story line and wonderful directio')
