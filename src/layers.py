import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tools

from . import misc
from . import scopes


# TODO: These don't seem to properly contribute to the 'total params' etc. model summaries
class ChainLayers(keras.layers.Layer):
    """Chains together multiple layers, e.g. f, g and h, so that the return value of this function may be called with an
    input i to in turn return f(g(h(i))).

    This is nearly the same as wrapping the layers in a Model - the difference is that that requires cluttering up the
    graph with many more tensors; this keeps things simpler. (And thus easier to keep track of in TensorBoard!)
    """

    def __init__(self, *layers, **kwargs):
        self.layers = tuple(layers)
        super(ChainLayers, self).__init__(**kwargs)

    # TODO: investigate why this throws an error. Possibly because of using crelu?
    # def build(self, input_shape):
    #     shape = input_shape
    #     for layer in self.layers:
    #         layer.build(shape)
    #         shape = layer.compute_output_shape(shape)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        x = input_shape
        for layer in self.layers:
            x = layer.compute_output_shape(x)
        return x

    def count_params(self):
        return sum(layer.count_params() for layer in self.layers)

    def get_config(self):
        config = super(ChainLayers, self).get_config()
        # TODO: support serializing and deserializing layers
        # config['layers'] = tuple(layer.get_config() for layer in self.layers)
        config['layers'] = self.layers
        return config

    def get_weights(self):
        return [weight for layer in self.layers for weight in layer.get_weights()]

    def set_weights(self, weights):
        for layer in self.layers:
            num_weights = len(layer.get_weights())
            layer_weights = weights[:num_weights]
            weights = weights[num_weights:]
            layer.set_weights(layer_weights)


def chain_layers(*layers):
    """Lightweight way to chain layers together. See also the ktools.ChainLayers layer, for wrapping them all up inside
    a full-blown Keras layer.
    """

    def chained_layers(inputs):
        x = inputs
        for layer in layers:
            x = layer(x)
        return x
    # To allow lookup later, if need be.
    chained_layers.layers = layers
    return chained_layers


def Periodize(kernel_size, data_format):
    """Periodizes the input with a margin suitable for the specified kernel_size."""

    margin_sizes = []
    for k_size in kernel_size:
        margin_sizes.append((int(np.floor((k_size - 1) / 2)), int(np.ceil((k_size - 1) / 2))))
    # no margin on the batch or channel dimensions
    if data_format == 'channels_last':
        margins = tuple([0, *margin_sizes, 0])
    else:  # channels_first
        margins = tuple([0, 0, *margin_sizes])
    return layers.Lambda(misc.periodize, name=misc.uniq_name('periodize'), arguments={'margin_size': margins})


def _make_periodic_conv(conv, dimension):
    # TODO: make this actually act like the convolutional layer, e.g. calling methods etc.
    @tools.rename(f'Periodic{conv.__name__}')
    def PeriodicConv(filters, kernel_size, strides=1, padding='periodic', data_format='channels_last', *args, **kwargs):
        f"""As the usual {conv}, but also supports the padding option 'periodic'."""

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dimension
        layer_list = []
        if padding == 'periodic':
            padding = 'valid'
            layer_list.append(Periodize(kernel_size, data_format))
        conv_layer = conv(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                          data_format=data_format, *args, **kwargs)
        layer_list.append(conv_layer)
        # TODO: switch to the true ChainLayers once that's working a little better
        chained_layers = chain_layers(*layer_list)
        chained_layers.conv_layer = conv_layer
        return chained_layers
    return PeriodicConv


PeriodicConv1D = _make_periodic_conv(layers.Conv1D, 1)
PeriodicConv2D = _make_periodic_conv(layers.Conv2D, 2)
PeriodicSeparableConv1D = _make_periodic_conv(layers.SeparableConv1D, 1)
PeriodicSeparableConv2D = _make_periodic_conv(layers.SeparableConv2D, 2)
PeriodicDepthwiseConv2D = _make_periodic_conv(layers.DepthwiseConv2D, 2)
PeriodicConv2DTranspose = _make_periodic_conv(layers.Conv2DTranspose, 2)
PeriodicConv3D = _make_periodic_conv(layers.Conv3D, 3)
PeriodicConv3DTranspose = _make_periodic_conv(layers.Conv3DTranspose, 3)


def _assert_equal_shape(o1, o2):
    return tools.assert_equal(o1, o2, getter=lambda x: x.shape.as_list(),
                              error_msg='{o1} and {o2} do not have equal shapes')


def _assert_equal_len(o1, o2):
    return tools.assert_equal(o1, o2, getter=lambda x: len(x), error_msg='{o1} and {o2} do not have equal lengths')


def replace_layers(model, new_layers, recursive=False):
    """Replaces multiple layers in a :model:. The argument :new_layers: should be a dictionary whose keys and values are
    both layers; the key corresponding to a layer in the :model: and the value corresponding to the layer that is
    replacing it.

    (The value only needs to be callable in the manner of the keras functional API, so for example it is acceptable to
    use the result of the chain_layers function as a value.)

    If :recursive: is True then layers which have layers in them will have replace_layers called on them (with
    recursive=True) in turn. It defaults to False.
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
                if recursive and isinstance(layer, keras.Model):
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
            with scopes.get_name_scope(node):
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


def dense_block(size, activation):
    """Creates a basic dense layer in the network: BatchNorm, then Activation, then Dense. The Activation will be
    :activation: and the Dense will have size :size: and no activation.
    """
    return chain_layers(layers.BatchNormalization(renorm=True),
                        layers.Activation(activation=activation),
                        layers.Dense(size))


def dense_change_size(size, activation):
    return layers.Dense(int(size[-1]), use_bias=False)


def residual_layers(make_block=dense_block, change_size=dense_change_size,
                    hidden_blocks=((512, 512), (512, 512), (512, 512), (512, 512)),
                    reshape='necessary', activation=tf.nn.crelu):
    """Returns a wrapper which creates a collection of residual layers when called on a tf.Tensor x.

    Arguments:
        make_block: A function taking two arguments, 'size' and 'activation', and returns a callable, such as a Keras
            Layer, which takes a tf.Tensor as input and returns a tf.Tensor as output. It will be called with each of
            the integers in 'hidden_blocks' as the 'size' argument, and 'activation' as the 'activation' argument.
        change_size: A function of the same form as make_block. Will be called whenever reshaping is necessary to
            perform the addition part of a ResNet. (Because the size of the output of the previous block is not the same
            size as the input to the block.) Unused in reshape='never'.
        hidden_blocks: A nested structure of tuples and integers, defining the sizes of the layers of the ResNet. If the
            entry of the tuple is an integer, then it corresponds to a layer of that size being created (via the
            make_block function). If the entry is a tuple then residual_layers is called recursively on the tuple, and
            the result added on in the manner of ResNets. i.e. setting hidden_blocks=(4, (3, 3)) would
            generate the following structure, with 'a' denoting the argument 'activation':

                                        --> make_block(3, a) -> make_block(3, a) --\
                                       /                                           v
            input -> make_block(4, a) -------------> change_size(3, a) ----------> add -> output

        reshape: One of 'always', 'necessary' or 'never'. If it is 'always' then change_size will always be called
            during the 'identity' part of the ResNet. If it is 'necessary' then it will be called only when necessary
            to perform the addition. If it is 'never' then change_size will not be used, and an Exception raised if the
            sizes do not allow addition.
        activation: The activation function to use throughout; passed as an argument to make_block and change_size.

    Returns:
        A tf.Tensor that is the output of the set of residual blocks.
    """

    if reshape not in ('always', 'necessary', 'never'):
        raise ValueError(f"reshape should be one of 'always', 'necessary' or 'never'.")

    def wrapper(x):
        for hidden_sizes in hidden_blocks:
            if isinstance(hidden_sizes, int):
                x = make_block(hidden_sizes, activation)(x)
            elif isinstance(hidden_sizes, (tuple, list)):
                y = x
                x = residual_layers(make_block, change_size, hidden_sizes, reshape, activation)(x)
                if reshape == 'always':
                    y = change_size(x.shape, activation)(y)
                elif reshape == 'necessary':
                    if x.shape[1:] != y.shape[1:]:  # remove batch dimension
                        y = change_size(x.shape, activation)(y)
                x = layers.Add()([x, y])
        return x
    return wrapper
