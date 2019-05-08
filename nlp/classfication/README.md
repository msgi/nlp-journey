## 文本分类算法

> 部分代码来自大牛的github: [textClassifier](https://github.com/jiangxinyang227/textClassifier), 感谢!

### 1) fasttext简单分类（速度快，可作为基准）

```python
train_path = 'data/imdb'
model = FastTextClassifier('model/fasttext/model')
model.predict('i don\'t like it because it is too boring')
model.predict('i like it ! its very interesting')
```


### 2) SVM算法

* 运用线性分类模型进行分类, 在文件svm_sklearn.py里.

```python
svm_model = SVMClassifier('model/svm/model.pkl')
```

### 3) BiLSTM

* 双向lstm获取句子的表示，采用简单的全连接层分类，简单的二分类

* 很容易过拟合

### 4) BiLSTM+Attention

* 双向lstm获取句子的表示，然后用attention机制，最后采用简单的全连接层分类，简单的二分类

```python
model = BiLSTMAttentionClassifier('GoogleNews-vectors-negative300.bin.gz', 'model/att','model/att/config.pkl', train=True)
print(model.predict_result('this is very good movie, i want to watch it again!'))
print(model.predict_result('this is very bad movie, i hate it'))
```

> 第一次训练的时候, 需要训练好的词向量，如google的， 后边再次训练的时候，train设置为True, 只用训练好的模型做推理的时候，train设置为False

* 实验结果

    | 算法             | 实验（1） | 实验（2） | 实验（3） | 实验（4） |
    | ---------------- | :-------- | --------- | --------- | --------- |
    | fasttext         |           |           |           |           |
    | svm              |           |           |           |           |
    | bilstm           |           |           |           |           |
    | bilstm+attention |           |           |           |           |

    