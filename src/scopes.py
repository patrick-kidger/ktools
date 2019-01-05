import tensorflow as tf


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


def get_current_scopes():
    """Gets the name of the current name scopes."""

    # We wrap this because it's in contrib; we'll need to change this once contrib is depreciated.
    return tf.contrib.framework.get_name_scope()
