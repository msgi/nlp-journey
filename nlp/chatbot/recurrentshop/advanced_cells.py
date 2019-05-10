from .cells import *


class RHNCell(ExtendedRNNCell):

    def __init__(self, units, recurrence_depth, **kwargs):
        self.recurrence_depth = recurrence_depth
        kwargs['units'] = units
        super(RHNCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        output_dim = self.output_dim
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        Rh = Dense(output_dim,
                   kernel_initializer=self.recurrent_initializer,
                   kernel_regularizer=self.recurrent_regularizer,
                   kernel_constraint=self.recurrent_constraint,
                   use_bias=self.use_bias,
                   bias_initializer=self.bias_initializer,
                   bias_regularizer=self.bias_regularizer,
                   bias_constraint=self.bias_constraint)

        Rt = Dense(output_dim,
                   kernel_initializer=self.recurrent_initializer,
                   kernel_regularizer=self.recurrent_regularizer,
                   kernel_constraint=self.recurrent_constraint,
                   use_bias=self.use_bias,
                   bias_initializer=self.bias_initializer,
                   bias_regularizer=self.bias_regularizer,
                   bias_constraint=self.bias_constraint)

        Wh = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   kernel_constraint=self.kernel_constraint,
                   use_bias=self.use_bias,
                   bias_initializer=self.bias_initializer,
                   bias_regularizer=self.bias_regularizer,
                   bias_constraint=self.bias_constraint)

        Wt = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   kernel_constraint=self.kernel_constraint,
                   use_bias=self.use_bias,
                   bias_initializer=self.bias_initializer,
                   bias_regularizer=self.bias_regularizer,
                   bias_constraint=self.bias_constraint)

        hl = add([Wh(x), Rh(h_tm1)])
        hl = Activation('tanh')(hl)

        tl = add([Wt(x), Rt(h_tm1)])
        tl = Activation('sigmoid')(tl)

        cl = Lambda(lambda x: 1.0 - x, output_shape=lambda s: s)(tl)
        cl = Activation('sigmoid')(cl)

        ht = add([multiply([hl, tl]), multiply([h_tm1, cl])])

        for _ in range(self.recurrence_depth - 1):
            hli = Dense(output_dim,
                        activation='tanh',
                        kernel_initializer=self.recurrent_initializer,
                        kernel_regularizer=self.recurrent_regularizer,
                        kernel_constraint=self.recurrent_constraint,
                        use_bias=self.use_bias,
                        bias_initializer=self.bias_initializer,
                        bias_regularizer=self.bias_regularizer,
                        bias_constraint=self.bias_constraint)(ht)
            tli = Dense(output_dim,
                        activation='sigmoid',
                        kernel_initializer=self.recurrent_initializer,
                        kernel_regularizer=self.recurrent_regularizer,
                        kernel_constraint=self.recurrent_constraint,
                        use_bias=self.use_bias,
                        bias_initializer=self.bias_initializer,
                        bias_regularizer=self.bias_regularizer,
                        bias_constraint=self.bias_constraint)(ht)

            cli = Lambda(lambda x: 1.0 - x, output_shape=lambda s: s)(tli)
            cli = Activation('sigmoid')(cli)
            ht = add([multiply([hli, tli]), multiply([ht, cli])])

        return Model([x, h_tm1], [ht, Identity()(ht)])
