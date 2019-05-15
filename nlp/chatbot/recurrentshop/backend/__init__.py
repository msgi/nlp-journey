from .tensorflow_backend import *

rnn = lambda *args, **kwargs: K.rnn(*args, **kwargs) + ([],)
