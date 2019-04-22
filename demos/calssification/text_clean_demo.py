from nlp.preprocess.clean_text import clean_en_text

if __name__ == '__main__':
    sentence = 'This is a good time\' , please be happy'

    print(clean_en_text(sentence))
