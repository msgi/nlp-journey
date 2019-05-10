from keras.models import Model
from keras.layers import *
from .engine import RNNCell


'''
This is a more readable version of cells.py.
'''


class SimpleRNNCell(RNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        h = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        h = Activation('tanh')(h)
        return Model([x, h_tm1], [h, h])


class GRUCell(RNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        z = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        z = Activation('sigmoid')(z)
        r = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        r = Activation('sigmoid')(r)
        h_prime = add([Dense(output_dim)(multiply([r, h_tm1])), Dense(output_dim, use_bias=False)(x)])
        h_prime = Activation('tanh')(h_prime)
        gate = Lambda(lambda x: x[0] * x[1] + (1. - x[0]) * x[2], output_shape=lambda s: s[0])
        h = gate([z, h_prime, h_tm1])
        return Model([x, h_tm1], [h, h])


class LSTMCell(RNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        f = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        f = Activation('sigmoid')(f)
        i = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        i = Activation('sigmoid')(i)
        c_prime = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        c_prime = Activation('tanh')(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation('tanh')(c)
        o = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        o = Activation('sigmoid')(o)
        h = multiply([o, c])
        return Model([x, h_tm1, c_tm1], [h, h, c])
