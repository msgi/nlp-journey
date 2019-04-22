from nlp.classfication.cnn_keras_classifier import CnnKerasClassifier

if __name__ == '__main__':
    # tf.app.run()
    cnn_classifier = CnnKerasClassifier(positive_data_file='../data/rt-polaritydata/rt-polarity.pos',
                                        negative_data_file='../data/rt-polaritydata/rt-polarity.neg')
    cnn_classifier.train()
