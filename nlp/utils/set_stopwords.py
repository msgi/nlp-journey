from nltk.corpus import stopwords


# 英文停用词直接引用nltk包里的即可
def set_en_stopwords():
    return set(stopwords.words('english'))


# 中文停用词自己实现
def set_zh_stopwords():
    raise NotImplementedError