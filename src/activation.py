import tensorflow as tf
import tools


def concat_activations(activation_funcs, funcname='concat_activations', default_scope_name='ConcatActivations'):
    """Creates a new activation function by concatenating the results of the given activation functions together.

    Be careful to distinguish this function from concat_activation.
    """

    @tools.rename(funcname)
    def concat(features, name=None, axis=-1, **kwargs):
        with tf.name_scope(name, default_scope_name, [features]) as name:
            features = tf.convert_to_tensor(features, name="features")
            activations = [activation_func(features, **kwargs) for activation_func in activation_funcs]
            return tf.concat(activations, axis, name=name)
    return concat


def minus_activation(activation_func, minus_name=None):
    """Creates a new activation function by negating the first argument given to the activation function.

    e.g. a ReLU would become the function x |-> max{0, -x}
    """

    if minus_name is None:
        minus_name = f'minus_{activation_func.__name__}'

    @tools.rename(minus_name)
    def minus(features, *args, **kwargs):
        return activation_func(-features, *args, **kwargs)
    return minus


def concat_activation(activation_func, concat_name=None, default_scope_name=None):
    """Creates a new activation function by using the given activation function twice, negating its input the second
    time, and concatenating the results.

    i.e. the manner in which a CReLU is created from a ReLU.

    Be careful to distinguish this function from concat_activations.
    """

    minus = minus_activation(activation_func)
    if concat_name is None:
        concat_name = f'concat_{activation_func.__name__}'
    if default_scope_name is None:
        default_scope_name = f"Concat{activation_func.__name__.capitalize()}"
    return concat_activations([activation_func, minus], concat_name, default_scope_name)


cleaky_relu = concat_activation(tf.nn.leaky_relu)
celu = concat_activation(tf.nn.elu)
cselu = concat_activation(tf.nn.selu)
