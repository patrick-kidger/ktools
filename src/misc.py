import functools as ft
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


_REENTERABLE_SCOPES = {}


# def reenterable_name_scope(name, *args, disallow_nonexistent=False, **kwargs):
#     try:
#         scope = _REENTERABLE_SCOPES[name]
#     except KeyError:
#         if disallow_nonexistent:
#             raise
#         scope = tf.name_scope(name, *args, **kwargs)
#         _REENTERABLE_SCOPES[name] = scope
#     else:
#         assert not args
#         assert not kwargs
#         scope = tf.name_scope(scope)
#     return scope


class reenterable_name_scope(tf.name_scope):
    _existing_scopes = {}

    def __init__(self, name, *args, disallow_nonexistent=False, **kwargs):
        try:
            name = self._existing_scopes[name]
        except KeyError as e:
            if disallow_nonexistent:
                raise ValueError(f'Sought reenterable name scope {name} does not exist.') from e
        super(reenterable_name_scope, self).__init__(name, *args, **kwargs)

    def __enter__(self):
        scope = super(reenterable_name_scope, self).__enter__()
        self._existing_scopes[self.name] = scope
        return scope


def get_name_scopes(node, same=False):
    """Gets the name scopes that this :node: was created in.

    A node is created every time a Keras layer is called; in particular through the functional API. Note that the layer
    itself does not provide enough information to determine this information: it can be called multiple times, in
    different scopes. Note that in this case some tensors internal to the layer could still be associated with whatever
    the first scope that called them is; Keras and scopes aren't really designed to play well with each other.

    This function allows one to resolve this issue partially: by getting the name scopes that a layer was once
    called in, it can be re-called in those scopes every time subsequently.

    Note that in order to get the previously existing scope, and not just with the same name as before (then made unique
    by adding a number), then the previous scopes must have been created with reenterable_name_scope, above. The default
    behaviour is to create a new scope with the same name; if :same: is True then it will seek a reenterable scope
    instead, and throw a ValueError if the reenterable scope does not exist.

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
    if same:
        scope_fn = ft.partial(reenterable_name_scope, disallow_nonexistent=True)
    else:
        scope_fn = tf.name_scope
    return tools.MultiWith([scope_fn(scope_name) for scope_name in scope_list])
