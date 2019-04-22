from nlp.classfication.cnn_classifier import train, preprocess


if __name__ == '__main__':
    # tf.app.run()
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)