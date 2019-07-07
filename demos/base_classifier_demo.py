import os
import sys
sys.path.append('/home/msg/workspace/pythons/nlp-journey')
from nlp.classfication.basic.basic_classifier import TextClassifier
if __name__ == '__main__':
    base_classifier = TextClassifier(model_path='./model/base/',
                                     config_path='./model/base/config.pkl',
                                     train=True, 
                                     vector_path='./data/GoogleNews-vectors-negative300.bin.gz')
    out = base_classifier.predict('this is very good movie, i want to watch it again!')
    print(out)
