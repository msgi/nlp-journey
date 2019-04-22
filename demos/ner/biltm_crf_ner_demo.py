from nlp.ner.bilstm_crf import NER

if __name__ == '__main__':
    ner = NER()
    ner.train()