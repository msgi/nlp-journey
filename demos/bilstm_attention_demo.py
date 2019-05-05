from nlp.classfication.bilstm_attention import BiLSTMAttentionModel
if __name__ == '__main__':
    model = BiLSTMAttentionModel(embedding_file='data/quora/GoogleNews-vectors-negative300.bin.gz')
    model.train()