import tensorflow as tf
import tensorflow.keras as keras
import tools

from . import misc


def chain_layers(*layers_, scope_name=None, reenterable_scope=False):
    """Chains together multiple layers, e.g. f, g and h, so that the return value of this function may be called with an
    input i to in turn return f(g(h(i))).

    This is very nearly the same as composing the given layers: keras.Model does composition, but then when it is
    subsequently used, it is treated as a single layer; this expands back out into the layers that were put into it.
    (Visible when calling the summary method, for example.)
    """

    def chained_layers(x):
        if scope_name is None:
            scope = tools.WithNothing()
        else:
            if reenterable_scope:
                scope = misc.reenterable_name_scope(scope_name)
            else:
                scope = tf.name_scope(scope_name)
        with scope:
            for layer in layers_:
                x = layer(x)
        return x
    return chained_layers


def _assert_equal_shape(o1, o2):
    return tools.assert_equal(o1, o2, getter=lambda x: x.shape.as_list(),
                              error_msg='{o1} and {o2} do not have equal shapes')


def _assert_equal_len(o1, o2):
    return tools.assert_equal(o1, o2, getter=lambda x: len(x), error_msg='{o1} and {o2} do not have equal lengths')


def replace_layers(model, new_layers, recursive=False, reenterable_scope=False):
    """Replaces multiple layers in a :model:. The argument :new_layers: should be a dictionary whose keys and values are
    both layers; the key corresponding to a layer in the :model: and the value corresponding to the layer that is
    replacing it.

    (The value only needs to be callable in the manner of the keras functional API, so for example it is acceptable to
    use the result of the chain_layers function as a value.)

    If :recursive: is True then layers which have layers in them will have replace_layers called on them (with
    recursive=True) in turn. It defaults to False.

    If :reenterable_scope: is True then the rebuilt model will try to use exactly the reenterable_scope scopes as
    before; it is False (the default) then it will instead create a scope with the 'reenterable_scope name', meaning
    that it will add a number to the end to make it unique, as usual. If :reenterable_scope: is True then all of the
    previous scopes used must be reenterable_name_scopes; if this is not the case then a ValueError will be thrown.
    """

    old_to_new = {k: k for k in model.inputs}  # The input tensors remain unchanged

    # For some odd reason model._nodes_by_depth is a dictionary with keys 0, 1, ...
    # noinspection PyProtectedMember
    for nodes_by_depth in (model._nodes_by_depth[i] for i in range(len(model._nodes_by_depth) - 1, -1, -1)):
        for node in nodes_by_depth:
            layer = node.outbound_layer
            if isinstance(layer, keras.layers.InputLayer):
                continue

            # Get the new layer
            try:
                new_layer = new_layers[layer]
            except KeyError:
                if recursive and hasattr(layer, 'layers'):
                    new_layer = replace_layers(layer, new_layers, recursive=True)
                else:
                    new_layer = layer

            # Now comes the tricky bit: replicating the argument signature the Layers were originally called with.
            # Keras saves some of this information, but doesn't seem to include all of the details: it saves the tensors
            # involved, but not the structure that they were called in.
            # Our assumptions, then:
            #  - A layer is only ever called with a Tensor, list of Tensors, or tuple of Tensors as its 'input' arg.
            #  - A layer only every returns a Tensor, list of Tensors, or tuple of Tensors.
            #  - A layer is not called with a length-one list or tuple of Tensors as its 'input' arg.
            # As 95% of use cases involve calling a Layer on a single Tensor, with another 4% being a list or tuple of
            # Tensors being passed to a Merge Layer, these assumptions aren't unreasonable: but things may unpredictably
            # break if these are not the case. (And we can't even necessarily detect that such a break has occurred
            # here, although we do try! Keras is surprisingly fragile.)

            new_input_tensors = [old_to_new[input_tensor] for input_tensor in node.input_tensors]
            if len(new_input_tensors) == 1:
                new_input_tensors = new_input_tensors[0]
            arguments = {} if node.arguments is None else node.arguments
            scope = misc.get_name_scopes(node, reenterable_scope=reenterable_scope)
            with scope:
                new_output_tensors = new_layer(new_input_tensors, **arguments)
            if isinstance(new_output_tensors, tf.Tensor):
                tools.assert_equal(len(node.output_tensors), 1, error_msg='{o1} does not have length 1')
                _assert_equal_shape(node.output_tensors[0], new_output_tensors)
                old_to_new[node.output_tensors[0]] = new_output_tensors
            elif isinstance(new_output_tensors, (list, tuple)):
                _assert_equal_len(node.output_tensors, new_output_tensors)
                for output_tensor, new_output_tensor in zip(node.output_tensors, new_output_tensors):
                    _assert_equal_shape(output_tensor.shape, new_output_tensor.shape)
                for output_tensor, new_output_tensor in zip(node.output_tensors, new_output_tensors):
                    old_to_new[output_tensor] = new_output_tensor
            else:
                raise RuntimeError(f'Layer {new_layer} (potentially replacing {layer}) returned something that is not a'
                                   f' Tensor, list of Tensors or tuple of Tensors: {new_output_tensors}. The Tensors '
                                   f'that the original layer returned are {node.output_tensors}.')

    return keras.Model(inputs=model.inputs, outputs=[old_to_new[o] for o in model.outputs])
