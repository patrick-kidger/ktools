import tensorflow as tf
import tensorflow.keras as keras
import tools


class WithTrainable:
    """A context in which every layer in :layers_: has their trainable flag set to :trainable:."""

    def __init__(self, layers, trainable=False):
        self.layers = layers
        self.trainable = trainable
        self.original_trainables = []

    def __enter__(self):
        for layer in self.layers:
            self.original_trainables.append(layer.trainable)
            layer.trainable = self.trainable

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer, original_trainable in zip(self.layers, self.original_trainables):
            layer.trainable = original_trainable


class TransformedSequence(keras.utils.Sequence):
    """Creates a new keras.utils.Sequence that applies a transformation to a previous one."""

    def __init__(self, sequence, transform):
        self.sequence = sequence
        self.transform = transform

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.transform(self.sequence[item])


def get_variable_scopes(node):
    """Gets the variable scopes that this :node: was created in.

    A node is created every time a Keras layer is called; in particular through the functional API. Note that the layer
    itself does not provide enough information to determine this information: it can be called multiple times, in
    different scopes. Note that in this case some tensors internal to the layer could still be associated with whatever
    the first scope that called them is; Keras and scopes aren't really designed to play well with each other.

    This function allows one to resolve this issue partially: by getting the variable scopes that a layer was once
    called in, it can be re-called in those scopes every time subsequently.

    Returns a context chaining together every scope that the :node: was called in.
    """

    layer_name = node.outbound_layer.name
    tensor_name = node.output_tensors[0].name
    tensor_name_parts = tensor_name.split('/')
    try:
        index = tensor_name_parts.index(layer_name)
    except ValueError as e:
        raise RuntimeError(f'Inconsistent tensor naming: node {node} has output layer {node.outbound_layer} with name '
                           f'{node.outbound_layer.name}, but produces a tensor {node.output_tensors[0]} with name '
                           f'{node.output_tensors[0].name}.') from e
    scope_list = tensor_name_parts[:index]
    return tools.MultiWith([tf.variable_scope(scope_name, reuse=True) for scope_name in scope_list])
