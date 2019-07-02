from nlp.classfication.text_cnn_classifier import TextCnnClassifier

if __name__ == '__main__':
    classifier = TextCnnClassifier('model/cnn',
                               config_file='model/cnn/config.pkl',
                               train=True,
                               pos_file='data/rt-polaritydata/rt-polarity.pos',
                               neg_file='data/rt-polaritydata/rt-polarity.neg')
