from nlp.classfication.cnn_classifier import CnnClassifier

if __name__ == '__main__':
    classifier = CnnClassifier('model/cnn',
                               train=True,
                               pos_file='data/rt-polaritydata/rt-polarity.pos',
                               neg_file='data/rt-polaritydata/rt-polarity.neg')
