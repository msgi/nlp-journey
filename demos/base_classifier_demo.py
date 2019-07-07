import sys
sys.path.append('e:/Workspace/pythons/nlp-journey')
from nlp.classfication.basic.basic_classifier import TextClassifier


if __name__ == '__main__':

    base_classifier = TextClassifier('model/base/','model/base/config.pkl', True)
    out = base_classifier.predict('this is very good movie, i want to watch it again!')
    print(out)