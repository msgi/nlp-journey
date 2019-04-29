from nlp.simililarity.siamese_similarity import SiameseSimilarity
if __name__ == '__main__':

    siamese = SiameseSimilarity('model/quora/siamese.h5','model/config.pkl',train=True,data_path='data/quora',embedding_file='')