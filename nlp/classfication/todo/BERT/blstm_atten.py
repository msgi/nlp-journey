import tensorflow as tf


# 构建模型
class BlstmAtten(object):
    """
    Bi-LSTM Attention 用于文本分类
    """
    def __init__(self, embedded_chars, hidden_sizes, labels, num_label,
                 dropout_rate, max_len):
        """
        构建BLSTM+ATTEN模型结构
        :param embedded_chars:
        :param hidden_sizes:
        :param labels:
        :param num_label:
        :param dropout_rate:
        :param max_len:
        """
        self.embedded_chars = embedded_chars
        self.hidden_sizes = hidden_sizes
        self.labels = labels
        self.num_label = num_label
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.embedding_size = embedded_chars.shape[-1].value

    def _blstm_atten(self):

        with tf.name_scope("embedding_dropout"):
            self.drop_embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(self.hidden_sizes):
                with tf.variable_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.drop_embedded_chars,
                                                                                   dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.drop_embedded_chars = tf.concat(outputs_, 2)

            # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.drop_embedded_chars, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            tf.logging.info("H shape: {}".format(H))
            output = self._attention(H)
            output_size = self.hidden_sizes[-1]

        return output, output_size

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hidden_size = self.hidden_sizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        new_m = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restore_m = tf.reshape(new_m, [-1, self.max_len])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restore_m)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.max_len, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        tf.logging.info("r shape: {}".format(r))
        sequeeze_r = tf.squeeze(r, axis=-1)
        tf.logging.info("sequeeze_r shape: {}".format(sequeeze_r))
        sentence_embedding = tf.tanh(sequeeze_r)
        tf.logging.info("sentence embedding shape: {}".format(sentence_embedding))
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentence_embedding, self.dropout_rate)

        return output

    def _output_layer(self, input_, input_shape):

        # 全连接层的输出
        with tf.variable_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[input_shape, self.num_label],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.get_variable("output_b",
                                       initializer=tf.constant(0.1, shape=[self.num_label]))
            tf.logging.info("output shape: {}".format(input_))
            tf.logging.info("output_w shape: {}".format(output_w))
            logits = tf.nn.xw_plus_b(input_, output_w, output_b, name="logits")

        return logits

    def _cal_loss(self, logits):
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            loss = tf.reduce_mean(losses)

        return loss

    def _get_prediction(self, logits):
        return tf.argmax(logits, -1, name="predictions")

    def gen_result(self):
        output, output_size = self._blstm_atten()
        logits = self._output_layer(output, output_size)
        loss = self._cal_loss(logits)

        predictions = self._get_prediction(logits)
        tf.logging.info("predictions: {}".format(predictions))

        return (loss, logits, predictions)