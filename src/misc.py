import tensorflow as tf
import tensorflow.keras as keras


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


def get_name_scope(node):
    """Gets the name scope that this :node: was created in.

    A node is created every time a Keras layer is called; in particular through the functional API. Note that the layer
    itself does not provide enough information to determine this information: it can be called multiple times, in
    different scopes. Note that in this case some tensors internal to the layer could still be associated with whatever
    the first scope that called them is; Keras and scopes aren't really designed to play well with each other.

    This function allows one to resolve this issue partially: by getting the name scope that a layer was once
    called in, it can be re-called in that scope every time subsequently.

    Returns a context chaining together every name scope that the :node: was called in.
    """

    layer_name = node.outbound_layer.name
    tensor_name = node.output_tensors[0].name
    index = tensor_name.index(layer_name)
    scope = tensor_name[:index]  # includes trailing slash to make this an absolute scope
    return tf.name_scope(scope)
