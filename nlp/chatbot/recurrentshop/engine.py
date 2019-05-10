from keras.layers import *
from keras.models import Model
from keras import initializers
from .backend import rnn, learning_phase_scope
from .generic_utils import serialize_function, deserialize_function
from keras.engine.base_layer import Node,_collect_previous_mask, _collect_input_shape
import inspect


if K.backend() == 'tensorflow':
    import tensorflow as tf

def _to_list(x):
    if type(x) is not list:
        x = [x]
    return x


class _OptionalInputPlaceHolder(Layer):

    def __init__(self, name=None, **kwargs):
        if not name:
            prefix = 'optional_input_placeholder'
            name = prefix + '_' + str(K.get_uid(prefix))
        kwargs['batch_input_shape'] = (2,)
        super(_OptionalInputPlaceHolder, self).__init__(**kwargs)
        self.tensor = K.zeros(shape=(2,))
        self.tensor._keras_shape = (2,)
        self.tensor._uses_learning_phase = False
        self.tensor._keras_history = (self, 0, 0)
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[],
             output_tensors=[self.tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[],
             output_shapes=[(2,)])
        self.build((2,))

    def call(self, inputs=None):
        return self.tensor


def _get_cells():
    from .cells import SimpleRNNCell, LSTMCell, GRUCell
    cells = {}
    cells['SimpleRNNCell'] = SimpleRNNCell
    cells['LSTMCell'] = LSTMCell
    cells['GRUCell'] = GRUCell
    cells['_OptionalInputPlaceHolder'] = _OptionalInputPlaceHolder
    return cells


def _is_rnn_cell(cell):
    return issubclass(cell.__class__, RNNCell)


def _is_all_none(iterable_or_element):
    if not isinstance(iterable_or_element, (list, tuple)):
        iterable = [iterable_or_element]
    else:
        iterable = iterable_or_element
    for element in iterable:
        if element is not None:
            return False
    return True


def _get_cell_input_shape(cell):
    if hasattr(cell, 'batch_input_shape'):
        cell_input_shape = cell.batch_input_shape
    elif hasattr(cell, 'input_shape'):
        cell_input_shape = cell.input_shape
    elif hasattr(cell, 'input_spec'):
        if isinstance(cell.input_spec, list):
            if hasattr(cell.input_spec[0], 'shape'):
                cell_input_shape = cell.input_spec[0].shape
            else:
                cell_input_shape = None
        else:
            if hasattr(cell.input_spec, 'shape'):
                cell_input_shape = cell.input_spec.shape
            else:
                cell_input_shape = None
    else:
        cell_input_shape = None

    if cell_input_shape is not None:
        if set(map(type, list(set(cell_input_shape) - set([None])))) != set([int]):
            cell_input_shape = cell_input_shape[0]

    return cell_input_shape


class RNNCell(Layer):

    def __init__(self, output_dim=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        if 'batch_input_shape' in kwargs:
            self.model = self.build_model(kwargs['batch_input_shape'])
        elif 'input_shape' in kwargs:
            self.model = self.build_model((None,) + kwargs['input_shape'])
        if not hasattr(self, 'input_ndim'):
            self.input_ndim = 2
        super(RNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if type(input_shape) is list:
            self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
            self.model = self.build_model(input_shape[0])
        else:
            self.model = self.build_model(input_shape)
            self.input_spec = [InputSpec(shape=shape) for shape in _to_list(self.model.input_shape)]

    def build_model(self, input_shape):
        raise Exception(NotImplemented)

    @property
    def num_states(self):
        if hasattr(self, 'model'):
            model = self.model
        else:
            model = self.build_model((None,) + (2,) * (self.input_ndim - 1))  # Don't judge. It was 3 in the morning.
        model_input = model.input
        if type(model_input) is list:
            return len(model_input[1:])
        else:
            return 0

    @property
    def state_shape(self):
        model_input = self.model.input
        if type(model_input) is list:
            if len(model_input) == 2:
                return K.int_shape(model_input[1])
            else:
                return list(map(K.int_shape, model_input[1:]))
        else:
            return None

    def compute_output_shape(self, input_shape):
        model_inputs = self.model.input
        if type(model_inputs) is list and type(input_shape) is not list:
            input_shape = [input_shape] + list(map(K.int_shape, self.model.input[1:]))
        return self.model.compute_output_shape(input_shape)

    def call(self, inputs, learning=None):
        return self.model.call(inputs)

    def get_layer(self, **kwargs):
        input_shape = self.model.input_shape
        if type(input_shape) is list:
            state_shapes = input_shape[1:]
            input_shape = input_shape[0]
        else:
            state_shapes = []
        input = Input(batch_shape=input_shape)
        initial_states = [Input(batch_shape=shape) for shape in state_shapes]
        output = self.model([input] + initial_states)
        if type(output) is list:
            final_states = output[1:]
            output = output[0]
        else:
            final_states = []
        return RecurrentModel(input=input, output=output, initial_states=initial_states, final_states=final_states, **kwargs)

    @property
    def updates(self):
        return self.model.updates

    def add_update(self, updates, inputs=None):
        self.model.add_update(updates, inputs)

    @property
    def uses_learning_phase(self):
        return self.model.uses_learning_phase

    @property
    def _per_input_losses(self):
        if hasattr(self, 'model'):
            return getattr(self.model, '_per_input_losses', {})
        else:
            return {}

    @_per_input_losses.setter
    def _per_input_losses(self, val):
        if hasattr(self, 'model'):
            self.model._per_input_losses = val

    @property
    def losses(self):
        if hasattr(self, 'model'):
            return self.model.losses
        else:
            return []

    @losses.setter
    def losses(self, val):
        if hasattr(self, 'model'):
            self.model.losses = val

    def add_loss(self, losses, inputs=None):
        self.model.add_loss(losses, inputs)

    @property
    def constraints(self):
        return self.model.constraints

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

    def get_losses_for(self, inputs):
        return self.model.get_losses_for(inputs)

    def get_updates_for(self, inputs):
        return self.model.get_updates_for(inputs)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(RNNCell, self).get_config()
        config.update(base_config)
        return config

    def compute_mask(self, inputs, mask=None):
        model_output = self.model.output
        if type(model_output) is list:
            return [None] * len(model_output)
        else:
            return None


class RNNCellFromModel(RNNCell):

    def __init__(self, model, **kwargs):
        self.model = model
        self.input_spec = [Input(batch_shape=shape) for shape in _to_list(model.input_shape)]
        self.build_model = lambda _: model
        super(RNNCellFromModel, self).__init__(batch_input_shape=model.input_shape, **kwargs)

    def get_config(self):
        config = super(RNNCellFromModel, self).get_config()
        if self.model is None:
            config['model_config'] = None
        else:
            config['model_config'] = self.model.get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects={}):
        if type(custom_objects) is list:
            custom_objects = {obj.__name__: obj for obj in custom_objects}
        custom_objects.update(_get_cells())
        model_config = config.pop('model_config')
        model = Model.from_config(model_config, custom_objects)
        return cls(model, **config)


class RecurrentModel(Recurrent):

    # INITIALIZATION

    def __init__(self, input, output, initial_states=None, final_states=None, readout_input=None, teacher_force=False, decode=False, output_length=None, return_states=False, state_initializer=None, **kwargs):
        inputs = [input]
        outputs = [output]
        state_spec = None
        if initial_states is not None:
            if type(initial_states) not in [list, tuple]:
                initial_states = [initial_states]
            state_spec = [InputSpec(shape=K.int_shape(state)) for state in initial_states]
            if final_states is None:
                raise Exception('Missing argument : final_states')
            else:
                self.states = [None] * len(initial_states)
            inputs += initial_states
        else:
            self.states = []
            state_spec = []

        if final_states is not None:
            if type(final_states) not in [list, tuple]:
                final_states = [final_states]
            assert len(initial_states) == len(final_states), 'initial_states and final_states should have same number of tensors.'
            if initial_states is None:
                raise Exception('Missing argument : initial_states')
            outputs += final_states
        self.decode = decode
        self.output_length = output_length
        if decode:
            if output_length is None:
                raise Exception('output_length should be specified for decoder')
            kwargs['return_sequences'] = True
        self.return_states = return_states
        if readout_input is not None:
            self.readout = True
            state_spec += [Input(batch_shape=K.int_shape(outputs[0]))]
            self.states += [None]
            inputs += [readout_input]
        else:
            self.readout = False
        if teacher_force and not self.readout:
            raise Exception('Readout should be enabled for teacher forcing.')
        self.teacher_force = teacher_force
        self.model = Model(inputs, outputs)
        super(RecurrentModel, self).__init__(**kwargs)
        input_shape = list(K.int_shape(input))
        if not decode:
            input_shape.insert(1, None)
        self.input_spec = InputSpec(shape=tuple(input_shape))
        self.state_spec = state_spec
        self._optional_input_placeholders = {}
        if state_initializer:
            if type(state_initializer) not in [list, tuple]:
                state_initializer = [state_initializer] * self.num_states
            else:
                state_initializer += [None] * (self.num_states - len(state_initializer))
            state_initializer = [initializers.get(init) if init else initializers.get('zeros') for init in state_initializer]
        self.state_initializer = state_initializer

    def build(self, input_shape):
        if type(input_shape) is list:
            input_shape = input_shape[0]
        if not self.decode:
            input_length = input_shape[1]
            if input_length is not None:
                input_shape = list(self.input_spec.shape)
                input_shape[1] = input_length
                input_shape = tuple(input_shape)
                self.input_spec = InputSpec(shape=input_shape)
        if type(self.model.input) is list:
            model_input_shape = self.model.input_shape[0]
        else:
            model_input_shape = self.model.input_shape
        if not self.decode:
            input_shape = input_shape[:1] + input_shape[2:]
        for i, j in zip(input_shape, model_input_shape):
            if i is not None and j is not None and i != j:
                raise Exception('Model expected input with shape ' + str(model_input_shape) +
                                '. Received input with shape ' + str(input_shape))
        if self.stateful:
            self.reset_states()
        self.built = True

    # STATES

    @property
    def num_states(self):
        model_input = self.model.input
        if type(model_input) is list:
            return len(model_input[1:])
        else:
            return 0

    def get_initial_state(self, inputs):
        if type(self.model.input) is not list:
            return []
        try:
            batch_size = K.int_shape(inputs)[0]
        except:
            batch_size = None
        state_shapes = list(map(K.int_shape, self.model.input[1:]))
        states = []
        if self.readout:
            state_shapes.pop()
            # default value for initial_readout is handled in call()
        for shape in state_shapes:
            if None in shape[1:]:
                raise Exception('Only the batch dimension of a state can be left unspecified. Got state with shape ' + str(shape))
            if shape[0] is None:
                ndim = K.ndim(inputs)
                z = K.zeros_like(inputs)
                slices = [slice(None)] + [0] * (ndim - 1)
                z = z[slices]  # (batch_size,)
                state_ndim = len(shape)
                z = K.reshape(z, (-1,) + (1,) * (state_ndim - 1))
                z = K.tile(z, (1,) + tuple(shape[1:]))
                states.append(z)
            else:
                states.append(K.zeros(shape))
        state_initializer = self.state_initializer
        if state_initializer:
            # some initializers don't accept symbolic shapes
            for i in range(len(state_shapes)):
                if state_shapes[i][0] is None:
                    if hasattr(self, 'batch_size'):
                        state_shapes[i] = (self.batch_size,) + state_shapes[i][1:]
                if None in state_shapes[i]:
                    state_shapes[i] = K.shape(states[i])
            num_state_init = len(state_initializer)
            num_state = self.num_states
            assert num_state_init == num_state, 'RNN has ' + str(num_state) + ' states, but was provided ' + str(num_state_init) + ' state initializers.'
            for i in range(len(states)):
                init = state_initializer[i]
                shape = state_shapes[i]
                try:
                    if not isinstance(init, initializers.Zeros):
                        states[i] = init(shape)
                except:
                    raise Exception('Seems the initializer ' + init.__class__.__name__ + ' does not support symbolic shapes(' + str(shape) + '). Try providing the full input shape (include batch dimension) for you RecurrentModel.')
        return states

    def reset_states(self, states_value=None):
        if len(self.states) == 0:
            return
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not hasattr(self, 'states') or self.states[0] is None:
            state_shapes = list(map(K.int_shape, self.model.input[1:]))
            self.states = list(map(K.zeros, state_shapes))

        if states_value is not None:
            if type(states_value) not in (list, tuple):
                states_value = [states_value] * len(self.states)
            assert len(states_value) == len(self.states), 'Your RNN has ' + str(len(self.states)) + ' states, but was provided ' + str(len(states_value)) + ' state values.'
            if 'numpy' not in type(states_value[0]):
                states_value = list(map(np.array, states_value))
            if states_value[0].shape == tuple():
                for state, val in zip(self.states, states_value):
                    K.set_value(state, K.get_value(state) * 0. + val)
            else:
                for state, val in zip(self.states, states_value):
                    K.set_value(state, val)
        else:
            if self.state_initializer:
                for state, init in zip(self.states, self.state_initializer):
                    if isinstance(init, initializers.Zeros):
                        K.set_value(state, 0 * K.get_value(state))
                    else:
                        K.set_value(state, K.eval(init(K.get_value(state).shape)))
            else:
                for state in self.states:
                    K.set_value(state, 0 * K.get_value(state))

    # EXECUTION

    def __call__(self, inputs, initial_state=None, initial_readout=None, ground_truth=None, **kwargs):
        req_num_inputs = 1 + self.num_states
        inputs = _to_list(inputs)
        inputs = inputs[:]
        if len(inputs) == 1:
            if initial_state is not None:
                if type(initial_state) is list:
                    inputs += initial_state
                else:
                    inputs.append(initial_state)
            else:
                if self.readout:
                    initial_state = self._get_optional_input_placeholder('initial_state', self.num_states - 1)
                else:
                    initial_state = self._get_optional_input_placeholder('initial_state', self.num_states)
                inputs += _to_list(initial_state)
            if self.readout:
                if initial_readout is None:
                    initial_readout = self._get_optional_input_placeholder('initial_readout')
                inputs.append(initial_readout)
            if self.teacher_force:
                req_num_inputs += 1
                if ground_truth is None:
                    ground_truth = self._get_optional_input_placeholder('ground_truth')
                inputs.append(ground_truth)
        assert len(inputs) == req_num_inputs, "Required " + str(req_num_inputs) + " inputs, received " + str(len(inputs)) + "."
        with K.name_scope(self.name):
            if not self.built:
                self.build(K.int_shape(inputs[0]))
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)
                    del self._initial_weights
                    self._initial_weights = None
            previous_mask = _collect_previous_mask(inputs[:1])
            user_kwargs = kwargs.copy()
            if not _is_all_none(previous_mask):
                if 'mask' in inspect.getargspec(self.call).args:
                    if 'mask' not in kwargs:
                        kwargs['mask'] = previous_mask
            input_shape = _collect_input_shape(inputs)
            output = self.call(inputs, **kwargs)
            output_mask = self.compute_mask(inputs[0], previous_mask)
            output_shape = self.compute_output_shape(input_shape[0])
            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output

    def call(self, inputs, initial_state=None, initial_readout=None, ground_truth=None, mask=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if type(mask) is list:
            mask = mask[0]
        if self.model is None:
            raise Exception('Empty RecurrentModel.')
        num_req_states = self.num_states
        if self.readout:
            num_actual_states = num_req_states - 1
        else:
            num_actual_states = num_req_states
        if type(inputs) is list:
            inputs_list = inputs[:]
            inputs = inputs_list.pop(0)
            initial_states = inputs_list[:num_actual_states]
            if len(initial_states) > 0:
                if self._is_optional_input_placeholder(initial_states[0]):
                    initial_states = self.get_initial_state(inputs)
            inputs_list = inputs_list[num_actual_states:]
            if self.readout:
                initial_readout = inputs_list.pop(0)
                if self.teacher_force:
                    ground_truth = inputs_list.pop()
        else:
            if initial_state is not None:
                if not isinstance(initial_state, (list, tuple)):
                    initial_states = [initial_state]
                else:
                    initial_states = list(initial_state)
                if self._is_optional_input_placeholder(initial_states[0]):
                    initial_states = self.get_initial_state(inputs)

            elif self.stateful:
                initial_states = self.states
            else:
                initial_states = self.get_initial_state(inputs)
        if self.readout:
            if initial_readout is None or self._is_optional_input_placeholder(initial_readout):
                output_shape = K.int_shape(_to_list((self.model.output))[0])
                output_ndim = len(output_shape)
                input_ndim = K.ndim(inputs)
                initial_readout = K.zeros_like(inputs)
                slices = [slice(None)] + [0] * (input_ndim - 1)
                initial_readout = initial_readout[slices]  # (batch_size,)
                initial_readout = K.reshape(initial_readout, (-1,) + (1,) * (output_ndim - 1))
                initial_readout = K.tile(initial_readout, (1,) + tuple(output_shape[1:]))
            initial_states.append(initial_readout)
            if self.teacher_force:
                if ground_truth is None or self._is_optional_input_placeholder(ground_truth):
                    raise Exception('ground_truth must be provided for RecurrentModel with teacher_force=True.')
                if K.backend() == 'tensorflow':
                    with tf.control_dependencies(None):
                        counter = K.zeros((1,))
                else:
                    counter = K.zeros((1,))
                counter = K.cast(counter, 'int32')  
                initial_states.insert(-1, counter)
                initial_states[-2]
                initial_states.insert(-1, ground_truth)
                num_req_states += 2
        if len(initial_states) != num_req_states:
            raise ValueError('Layer requires ' + str(num_req_states) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        preprocessed_input = self.preprocess_input(inputs, training=None)
        constants = self.get_constants(inputs, training=None)
        if self.decode:
            initial_states.insert(0, inputs)
            preprocessed_input = K.zeros((1, self.output_length, 1))
            input_length = self.output_length
        else:
            input_length = input_shape[1]
        if self.uses_learning_phase:
            with learning_phase_scope(0):
                last_output_test, outputs_test, states_test, updates = rnn(self.step,
                                                                           preprocessed_input,
                                                                           initial_states,
                                                                           go_backwards=self.go_backwards,
                                                                           mask=mask,
                                                                           constants=constants,
                                                                           unroll=self.unroll,
                                                                           input_length=input_length)
            with learning_phase_scope(1):
                last_output_train, outputs_train, states_train, updates = rnn(self.step,
                                                                              preprocessed_input,
                                                                              initial_states,
                                                                              go_backwards=self.go_backwards,
                                                                              mask=mask,
                                                                              constants=constants,
                                                                              unroll=self.unroll,
                                                                              input_length=input_length)

            last_output = K.in_train_phase(last_output_train, last_output_test, training=training)
            outputs = K.in_train_phase(outputs_train, outputs_test, training=training)
            states = []
            for state_train, state_test in zip(states_train, states_test):
                states.append(K.in_train_phase(state_train, state_test, training=training))

        else:
            last_output, outputs, states, updates = rnn(self.step,
                                                        preprocessed_input,
                                                        initial_states,
                                                        go_backwards=self.go_backwards,
                                                        mask=mask,
                                                        constants=constants,
                                                        unroll=self.unroll,
                                                        input_length=input_length)
        states = list(states)
        if self.decode:
            states.pop(0)
        if self.readout:
            states.pop()
            if self.teacher_force:
                states.pop()
                states.pop()
        if len(updates) > 0:
            self.add_update(updates)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            y = outputs
        else:
            y = last_output
        if self.return_states:
            return [y] + states
        else:
            return y

    def step(self, inputs, states):
        states = list(states)
        if self.teacher_force:
            readout = states.pop()
            ground_truth = states.pop()
            assert K.ndim(ground_truth) == 3, K.ndim(ground_truth)
            counter = states.pop()
            if K.backend() == 'tensorflow':
                with tf.control_dependencies(None):
                    zero = K.cast(K.zeros((1,))[0], 'int32')
                    one = K.cast(K.zeros((1,))[0], 'int32')
            else:
                zero = K.cast(K.zeros((1,))[0], 'int32')
                one = K.cast(K.zeros((1,))[0], 'int32')
            slices = [slice(None), counter[0] - K.switch(counter[0], one, zero)] + [slice(None)] * (K.ndim(ground_truth) - 2)
            ground_truth_slice = ground_truth[slices]
            readout = K.in_train_phase(K.switch(counter[0], ground_truth_slice, readout), readout)
            states.append(readout)
        if self.decode:
            model_input = states
        else:
            model_input = [inputs] + states
        shapes = []
        for x in model_input:
            if hasattr(x, '_keras_shape'):
                shapes.append(x._keras_shape)
                del x._keras_shape  # Else keras internals will get messed up.
        model_output = _to_list(self.model.call(model_input))
        for x, s in zip(model_input, shapes):
            setattr(x, '_keras_shape', s)
        if self.decode:
            model_output.insert(1, model_input[0])
        for tensor in model_output:
            tensor._uses_learning_phase = self.uses_learning_phase
        states = model_output[1:]
        output = model_output[0]
        if self.readout:
            states += [output]
            if self.teacher_force:
                states.insert(-1, counter + 1)
                states.insert(-1, ground_truth)
        return output, states

    # SHAPE, MASK, WEIGHTS

    def compute_output_shape(self, input_shape):
        if not self.decode:
            if type(input_shape) is list:
                input_shape[0] = self._remove_time_dim(input_shape[0])
            else:
                input_shape = self._remove_time_dim(input_shape)
        input_shape = _to_list(input_shape)
        input_shape = [input_shape[0]] + [K.int_shape(state) for state in self.model.input[1:]]
        output_shape = self.model.compute_output_shape(input_shape)
        if type(output_shape) is list:
            output_shape = output_shape[0]
        if self.return_sequences:
            if self.decode:
                output_shape = output_shape[:1] + (self.output_length,) + output_shape[1:]
            else:
                output_shape = output_shape[:1] + (self.input_spec.shape[1],) + output_shape[1:]
        if self.return_states and len(self.states) > 0:
            output_shape = [output_shape] + list(map(K.int_shape, self.model.output[1:]))
        return output_shape

    def compute_mask(self, input, input_mask=None):
        mask = input_mask[0] if type(input_mask) is list else input_mask
        mask = mask if self.return_sequences else None
        mask = [mask] + [None] * len(self.states) if self.return_states else mask
        return mask

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    # LAYER ATTRIBS

    @property
    def updates(self):
        return self.model.updates

    def add_update(self, updates, inputs=None):
        self.model.add_update(updates, inputs)

    @property
    def uses_learning_phase(self):
        return self.teacher_force or self.model.uses_learning_phase

    @property
    def _per_input_losses(self):
        if hasattr(self, 'model'):
            return getattr(self.model, '_per_input_losses', {})
        else:
            return {}

    @_per_input_losses.setter
    def _per_input_losses(self, val):
        if hasattr(self, 'model'):
            self.model._per_input_losses = val

    @property
    def losses(self):
        if hasattr(self, 'model'):
            return self.model.losses
        else:
            return []

    @losses.setter
    def losses(self, val):
        if hasattr(self, 'model'):
            self.model.losses = val

    def add_loss(self, losses, inputs=None):
        self.model.add_loss(losses, inputs)

    @property
    def constraints(self):
        return self.model.constraints

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

    def get_losses_for(self, inputs):
        return self.model.get_losses_for(inputs)

    def get_updates_for(self, inputs):
        return self.model.get_updates_for(inputs)

    def _remove_time_dim(self, shape):
        return shape[:1] + shape[2:]

    # SERIALIZATION

    def _serialize_state_initializer(self):
        si = self.state_initializer
        if si is None:
            return None
        elif type(si) is list:
            return list(map(initializers.serialize, si))
        else:
            return initializers.serialize(si)

    def get_config(self):
        config = {'model_config': self.model.get_config(),
                  'decode': self.decode,
                  'output_length': self.output_length,
                  'return_states': self.return_states,
                  'state_initializer': self._serialize_state_initializer()            
                  }
        base_config = super(RecurrentModel, self).get_config()
        config.update(base_config)
        return config

    @classmethod
    def from_config(cls, config, custom_objects={}):
        if type(custom_objects) is list:
            custom_objects = {obj.__name__: obj for obj in custom_objects}
        custom_objects.update(_get_cells())
        config = config.copy()
        model_config = config.pop('model_config')
        if model_config is None:
            model = None
        else:
            model = Model.from_config(model_config, custom_objects)
        if type(model.input) is list:
            input = model.input[0]
            initial_states = model.input[1:]
        else:
            input = model.input
            initial_states = None
        if type(model.output) is list:
            output = model.output[0]
            final_states = model.output[1:]
        else:
            output = model.output
            final_states = None
        return cls(input, output, initial_states, final_states, **config)

    def get_cell(self, **kwargs):
        return RNNCellFromModel(self.model, **kwargs)

    def _get_optional_input_placeholder(self, name=None, num=1):
        if name:
            if name not in self._optional_input_placeholders:
                if num > 1:
                    self._optional_input_placeholders[name] = [self._get_optional_input_placeholder() for _ in range(num)]
                else:
                    self._optional_input_placeholders[name] = self._get_optional_input_placeholder()
            return self._optional_input_placeholders[name]
        if num == 1:
            optional_input_placeholder = _to_list(_OptionalInputPlaceHolder()._inbound_nodes[0].output_tensors)[0]
            assert self._is_optional_input_placeholder(optional_input_placeholder)
            return optional_input_placeholder
        else:
            y = []
            for _ in range(num):
                optional_input_placeholder = _to_list(_OptionalInputPlaceHolder()._inbound_nodes[0].output_tensors)[0]
                assert self._is_optional_input_placeholder(optional_input_placeholder)
                y.append(optional_input_placeholder)
            return y

    def _is_optional_input_placeholder(self, x):
        if hasattr(x, '_keras_history'):
            if isinstance(x._keras_history[0], _OptionalInputPlaceHolder):
                return True
        return False


class RecurrentSequential(RecurrentModel):

    def __init__(self, state_sync=False, decode=False, output_length=None, return_states=False, readout=False, readout_activation='linear', teacher_force=False, state_initializer=None, **kwargs):
        self.state_sync = state_sync
        self.cells = []
        if decode and output_length is None:
            raise Exception('output_length should be specified for decoder')
        self.decode = decode
        self.output_length = output_length
        if decode:
            if output_length is None:
                raise Exception('output_length should be specified for decoder')
            kwargs['return_sequences'] = True
        self.return_states = return_states
        super(RecurrentModel, self).__init__(**kwargs)
        self.readout = readout
        self.readout_activation = activations.get(readout_activation)
        self.teacher_force = teacher_force
        self._optional_input_placeholders = {}
        if state_initializer:
            if type(state_initializer) in [list, tuple]:
                state_initializer = [initializers.get(init) if init else initializers.get('zeros') for init in state_initializer]
            else:
                state_initializer = initializers.get(state_initializer)
        self._state_initializer = state_initializer

    @property
    def state_initializer(self):
        if self._state_initializer is None:
            return None
        elif type(self._state_initializer) is list:
            return self._state_initializer + [initializers.get('zeros')] * (self.num_states - len(self._state_initializer))
        else:
            return [self._state_initializer] * self.num_states

    @state_initializer.setter
    def state_initializer(self, value):
        self._state_initializer = value

    @property
    def num_states(self):
        if hasattr(self, 'model'):
            return super(RecurrentSequential, self).num_states
        num = 0
        for cell in self.cells:
            if _is_rnn_cell(cell):
                num += cell.num_states
                if self.state_sync:
                    break
        if self.readout:
            num += 1
        return num

    def add(self, cell):
        self.cells.append(cell)
        cell_input_shape = _get_cell_input_shape(cell)
        if len(self.cells) == 1:
            if len(self.cells) == 1:
                if self.decode:
                    self.input_spec = InputSpec(shape=cell_input_shape)
                else:
                    self.input_spec = InputSpec(shape=cell_input_shape[:1] + (None,) + cell_input_shape[1:])

        if cell_input_shape is not None:
            cell_input_shape = cell.batch_input_shape
            batch_size = cell_input_shape[0]
            if batch_size is not None:
                self.batch_size = batch_size
            if not self.stateful:
                self.states = [None] * self.num_states

    def build(self, input_shape):
        if hasattr(self, 'model'):
            del self.model
        # Try and get batch size for initializer
        if not hasattr(self, 'batch_size'):
            if hasattr(self, 'batch_input_shape'):
                batch_size = self.batch_input_shape[0]
                if batch_size is not None:
                    self.batch_size = batch_size
        if self.state_sync:
            if type(input_shape) is list:
                x_shape = input_shape[0]
                if not self.decode:
                    input_length = x_shape.pop(1)
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                input = Input(batch_shape=x_shape)
                initial_states = [Input(batch_shape=shape) for shape in input_shape[1:]]
            else:
                if not self.decode:
                    input_length = input_shape[1]
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                    input = Input(batch_shape=input_shape[:1] + input_shape[2:])
                else:
                    input = Input(batch_shape=input_shape)
                initial_states = []
            output = input
            final_states = initial_states[:]
            for cell in self.cells:
                if _is_rnn_cell(cell):
                    if not initial_states:
                        cell.build(K.int_shape(output))
                        initial_states = [Input(batch_shape=shape) for shape in _to_list(cell.state_shape)]
                        final_states = initial_states[:]
                    cell_out = cell([output] + final_states)
                    if type(cell_out) is not list:
                        cell_out = [cell_out]
                    output = cell_out[0]
                    final_states = cell_out[1:]
                else:
                    output = cell(output)
        else:
            if type(input_shape) is list:
                x_shape = input_shape[0]
                if not self.decode:
                    input_length = x_shape.pop(1)
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                input = Input(batch_shape=x_shape)
                initial_states = [Input(batch_shape=shape) for shape in input_shape[1:]]
                output = input
                final_states = []
                for cell in self.cells:
                    if _is_rnn_cell(cell):
                        cell_initial_states = initial_states[len(final_states): len(final_states) + cell.num_states]
                        cell_in = [output] + cell_initial_states
                        cell_out = _to_list(cell(cell_in))
                        output = cell_out[0]
                        final_states += cell_out[1:]
                    else:
                        output = cell(output)
            else:
                if not self.decode:
                    input_length = input_shape[1]
                    if input_length is not None:
                        shape = list(self.input_spec.shape)
                        shape[1] = input_length
                        self.input_spec = InputSpec(shape=tuple(shape))
                    input = Input(batch_shape=input_shape[:1] + input_shape[2:])
                else:
                    input = Input(batch_shape=input_shape)
                output = input
                initial_states = []
                final_states = []
                for cell in self.cells:
                    if _is_rnn_cell(cell):
                        cell.build(K.int_shape(output))
                        state_inputs = [Input(batch_shape=shape) for shape in _to_list(cell.state_shape)]
                        initial_states += state_inputs
                        cell_in = [output] + state_inputs
                        cell_out = _to_list(cell(cell_in))
                        output = cell_out[0]
                        final_states += cell_out[1:]
                    else:
                        output = cell(output)

        self.model = Model([input] + initial_states, [output] + final_states)
        self.states = [None] * len(initial_states)
        if self.readout:
            readout_input = Input(batch_shape=K.int_shape(output), name='readout_input')
            if self.readout_activation.__name__ == 'linear':
                readout = Lambda(lambda x: x + 0., output_shape=lambda s: s)(readout_input)
            else:
                readout = Activation(self.readout_activation)(readout_input)
            input = Input(batch_shape=K.int_shape(input))
            if self.readout in [True, 'add']:
                input_readout_merged = add([input, readout])
            elif self.readout in ['mul', 'multiply']:
                input_readout_merged = multiply([input, readout])
            elif self.readout in ['avg', 'average']:
                input_readout_merged = average([input, readout])
            elif self.readout in ['max', 'maximum']:
                input_readout_merged = maximum([input, readout])
            elif self.readout == 'readout_only':
                input_readout_merged = readout
            initial_states = [Input(batch_shape=K.int_shape(s)) for s in initial_states]
            output = _to_list(self.model([input_readout_merged] + initial_states))
            final_states = output[1:]
            output = output[0]
            self.model = Model([input] + initial_states + [readout_input], [output] + final_states)
            self.states.append(None)
        super(RecurrentSequential, self).build(input_shape)

    def get_config(self):
        config = {'cells': list(map(serialize, self.cells)),
                  'decode': self.decode,
                  'output_length': self.output_length,
                  'readout': self.readout,
                  'teacher_force': self.teacher_force,
                  'return_states': self.return_states,
                  'state_sync': self.state_sync,
                  'state_initializer': self._serialize_state_initializer(),
                  'readout_activation': activations.serialize(self.readout_activation)}
        base_config = super(RecurrentModel, self).get_config()
        config.update(base_config)
        return config

    @classmethod
    def from_config(cls, config, custom_objects={}):
        custom_objects.update(_get_cells())
        cells = config.pop('cells')
        rs = cls(**config)
        for cell_config in cells:
            cell = deserialize(cell_config, custom_objects)
            rs.add(cell)
        return rs


# Legacy
RecurrentContainer = RecurrentSequential
