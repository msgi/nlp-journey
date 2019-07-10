from nlp.ner.bilstm_crf import BiLSTMCRFNamedEntityRecognition

if __name__ == '__main__':
    ner = BiLSTMCRFNamedEntityRecognition('model/ner/crf.h5', 'model/ner/config.pkl', train=True, file_path='data/ner')
    ner.predict('中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚')
