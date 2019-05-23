### 激活函数

**ReLU、Leaky ReLU、PReLU和RReLU的比较**

[几种激活函数的比较](https://blog.csdn.net/guorongronghe/article/details/70174476)
[激活函数ReLU、Leaky ReLU、PReLU和RReLU](https://blog.csdn.net/qq_23304241/article/details/80300149)

![relu](../image/basic/relu.jpeg)

* PReLU中的ai是根据数据变化的；
* Leaky ReLU中的ai是固定的；
* RReLU中的aji是一个在一个给定的范围内随机抽取的值，这个值在测试环节就会固定下来。

### BatchNorm

[博客](https://www.cnblogs.com/guoyaohua/p/8724433.html)

> IID独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障. BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。

### optimizer

[optimizer的比较](https://blog.csdn.net/weixin_40170902/article/details/80092628)


#### 梯度下降法(Gradient Descent)

**标准梯度下降法(GD)**

![gd](../image/basic/gd.png)

**批量梯度下降法(BGD)**

![bgd](../image/basic/bgd.png)

**随机梯度下降法(SGD)**

![sgd](../image/basic/sgd.png)

![gt](../image/basic/gt.png)

```python
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
```

#### 动量优化法

**Momentum**

![momentum](../image/basic/momentum.png)

α一般取0.9(表示最大速度10倍于SGD)

```python
train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
```

**牛顿加速梯度（NAG, Nesterov accelerated gradient）算法)**

![nag](../image/basic/nag.png)

```python
train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)
```

#### 自适应学习率优化算法

**AdaGrad算法**

![adagrad](../image/basic/adagrad.png)

假定一个多分类问题，i表示第i个分类，t表示第t迭代同时也表示分类i累计出现的次数。η0表示初始的学习率取值一般为0.01，ϵ是一个取值很小的数（一般为1e-8）为了避免分母为0。Wt表示t时刻即第t迭代模型的参数，gt,i=ΔJ(Wt,i)表示t时刻，指定分类i，代价函数J(⋅)关于W的梯度。

Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；缺点在于，随着迭代次数增多，学习率会越来越小，最终会趋近于0。

```python
train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
```

**RMSProp算法**

![RMSProp](../image/basic/rmsprop.png)

gt=ΔJ(Wt)表示t次迭代代价函数关于W的梯度大小. ![e](../image/basic/e.png)表示前t次的梯度平方的均值。

```python
train_step = tf.train.RMSPropOptimizer(0.01).minimize(loss)
```

**AdaDelta算法**

AdaGrad算法和RMSProp算法都需要指定全局学习率，AdaDelta算法结合两种算法每次参数的更新步长即

![AdaDelta算法](../image/basic/adadelta.png)

![AdaDelta算法](../image/basic/adadelta1.png)

```python
train_step = tf.train.AdadeltaOptimizer(1).minimize(loss)
```

**adam**

![adam](../image/basic/adam.png)

```python
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
```

目前，最流行并且使用很高的优化器（算法）包括SGD、具有动量的SGD、RMSprop、具有动量的RMSProp、AdaDelta和Adam。在实际应用中，选择哪种优化器应结合具体问题；同时，也优化器的选择也取决于使用者对优化器的熟悉程度（比如参数的调节等等）。


### 正则化方法

[L1和L2 regularization、数据集扩增、dropout](https://blog.csdn.net/u010402786/article/details/49592239)

```python
import keras
keras.layers.core.Dense(
    units, #代表该层的输出维度
    activation=None, #激活函数.但是默认 liner
    use_bias=True, #是否使用b
    kernel_initializer='glorot_uniform', #初始化w权重，keras/initializers.py
    bias_initializer='zeros', #初始化b权重
    kernel_regularizer=None, #施加在权重w上的正则项,keras/regularizer.py
    bias_regularizer=None, #施加在偏置向量b上的正则项
    activity_regularizer=None, #施加在输出上的正则项
    kernel_constraint=None, #施加在权重w上的约束项
    bias_constraint=None #施加在偏置b上的约束项
)
```