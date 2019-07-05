# 常见问题

> 整理、收集自网上资源

## 1. 使用cuda出现问题的解决方式

```bash
sudo ldconfig /usr/local/cuda-9.0/lib64
sudo ln -sf /usr/local/cuda-9.0/lib64/libcudnn.so.7.0.5 /usr/local/cuda-9.0/lib64/libcudnn.so.7
```

## 2. 找不到自己写的模块

**在代码入口处加入一下代码：**

```python
import sys
sys.path.append('/project_path/module')
```

## 3. CNN的feature map计算方式

![cnn](../images/cnn_feature_map.png)

## 4. RNN cell如何计算隐藏状态？

![rnn_cell](../images/rnn_cell.gif)

## 5. LSTM中各种门的作用？

LSTM 的核心概念在于**细胞状态**以及“**门**”结构。细胞状态相当于信息传输的路径，让信息能在序列连中传递下去。可以将其看作网络的“记忆”。理论上讲，细胞状态能够将序列处理过程中的相关信息一直传递下去。

LSTM 有三种类型的门结构：遗忘门、输入门和输出门。

**遗忘门**

遗忘门用于决定应丢弃或保留哪些信息。来自前一个隐藏状态的信息和当前输入的信息同时传递到 sigmoid 函数中去，输出值介于 0 和 1 之间，越接近 0 意味着越应该丢弃，越接近 1 意味着越应该保留。

![forget_gate](../images/lstm/forget_gate.gif)

**输入门**

输入门用于更新细胞状态。首先将前一层隐藏状态的信息和当前输入的信息传递到 sigmoid 函数中去。将值调整到 0~1 之间来决定要更新哪些信息。0 表示不重要，1 表示重要。

其次还要将前一层隐藏状态的信息和当前输入的信息传递到 tanh 函数中去，创造一个新的侯选值向量。最后将 sigmoid 的输出值与 tanh 的输出值相乘，sigmoid 的输出值将决定 tanh 的输出值中哪些信息是重要且需要保留下来的。

![input_gate](../images/lstm/input_gate.gif)

**细胞状态**

下一步，就是计算细胞状态。首先前一层的细胞状态与遗忘向量逐点相乘。如果它乘以接近 0 的值，意味着在新的细胞状态中，这些信息是需要丢弃掉的。然后再将该值与输入门的输出值逐点相加，将神经网络发现的新信息更新到细胞状态中去。至此，就得到了更新后的细胞状态。

![cell_state](../images/lstm/cell_state.gif)

**输出门**

输出门用来确定下一个隐藏状态的值。首先，我们将前一个隐藏状态和当前输入传递到 sigmoid 函数中，然后将新得到的细胞状态传递给 tanh 函数。

最后将 tanh 的输出与 sigmoid 的输出相乘，以确定隐藏状态应携带的信息。再将隐藏状态作为当前细胞的输出，把新的细胞状态和新的隐藏状态传递到下一个时间步长中去。

![output_gate](../images/lstm/output_gate.gif)