import keras.backend as K


if K.backend() == 'tensorflow':
	from .tensorflow_backend import *
	rnn = lambda *args, **kwargs: K.rnn(*args, **kwargs) + ([],)
elif K.backend() == 'theano':
	from .theano_backend import *
else:
	raise Exception(K.backend() + ' backend is not supported.')
