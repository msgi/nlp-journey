import sys
from nlp.classfication.dl.basic_classifier import TextClassifier

sys.path.append('/home/msg/workspace/pythons/nlp-journey')

if __name__ == '__main__':
    base_classifier = TextClassifier(model_path='./model/base/',
                                     config_path='./model/base/config.pkl',
                                     train=False,
                                     vector_path='./data/GoogleNews-vectors-negative300.bin.gz')
    out = base_classifier.predict(
        ['this is very good movie , i want to watch it again!', 'this is very bad movie , i hate it!'])
    out2 = base_classifier.predict('this is very good movie , i want to watch it again!')
    print(out)
    print(out2)
