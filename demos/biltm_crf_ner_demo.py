from nlp.ner.bilstm_crf import BiLSTMNamedEntityRecognition

if __name__ == '__main__':
    ner = BiLSTMNamedEntityRecognition()
    ner.train()