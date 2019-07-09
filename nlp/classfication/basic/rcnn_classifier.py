from keras import Input, Model
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Dense

from .basic_classifier import TextClassifier


class TextRCNNClassifier(TextClassifier):
    """
        RCNN 整体的模型构建流程如下：

    　　1）利用Bi-LSTM获得上下文的信息，类似于语言模型。

    　　2）将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]。

    　　3）将拼接后的向量非线性映射到低维。

    　　4）向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向量，该过程类似于max-pool。

    　　5）softmax分类。
    """

    def build_model(self):
        inputs = Input((self.maxlen,))
        embedding = Embedding(len(self.embeddings),
                              300,
                              weights=[self.embeddings],
                              trainable=False)(inputs)
        x_context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])
        cs = []
        for kernel_size in range(1, 5):
            c = Conv1D(128, kernel_size, activation='relu')(x)
            cs.append(c)
        pools = [GlobalAveragePooling1D()(conv) for conv in cs] + [GlobalMaxPooling1D()(c) for c in cs]
        x = Concatenate()(pools)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, batch_size=512, epochs=20):
        super(TextRCNNClassifier, self).train(128, 2)
