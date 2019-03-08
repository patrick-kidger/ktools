import tensorflow as tf
import tensorflow.keras.activations as activations
import tensorflow.keras.constraints as constraints
import tensorflow.keras.initializers as init
import tensorflow.keras.layers as layers
import tools


def concat_multiple_activations(activation_funcs, funcname='concat_activations',
                                default_scope_name='ConcatActivations'):
    """Creates a new activation function by concatenating the results of the given activation functions together."""

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
    return concat_multiple_activations([activation_func, minus], concat_name, default_scope_name)


def softthresh(features, tau=1.0, name=None):
    with tf.name_scope(name, "softthresh", [features, tau]) as name:
        features = tf.convert_to_tensor(features, name="features")
        minus_tau = tf.convert_to_tensor(-tau, dtype=features.dtype, name="minustau")
        pos = tf.nn.relu(minus_tau + features, name="pos")
        neg = tf.nn.relu(minus_tau - features, name="neg")
        return tf.subtract(pos, neg, name=name)


cleaky_relu = concat_activation(tf.nn.leaky_relu)
celu = concat_activation(tf.nn.elu)
cselu = concat_activation(tf.nn.selu)
# No csoftthresh, ctanh etc. because they're odd functions

# Awful hackiness to make Keras deserialization work
tools.safe_add(activations.get.__globals__, 'identity', tf.identity)
tools.safe_add(activations.get.__globals__, 'softthresh', softthresh)
tools.safe_add(activations.get.__globals__, 'crelu', tf.nn.crelu)
tools.safe_add(activations.get.__globals__, 'cleaky_relu', cleaky_relu)
tools.safe_add(activations.get.__globals__, 'celu', celu)
tools.safe_add(activations.get.__globals__, 'cselu', cselu)


class QuasiIdentity(layers.PReLU):
    def __init__(self, activation_func, alpha_initializer=init.Constant(0.5), alpha_regularizer=None, **kwargs):
        self.activation_func = activations.get(activation_func)
        if alpha_regularizer == 'convex':
            kwargs['alpha_regularizer'] = constraints.MinMaxNorm(0, 1)
        super(QuasiIdentity, self).__init__(alpha_initializer=alpha_initializer, alpha_regularizer=alpha_regularizer,
                                            **kwargs)

    def call(self, inputs, mask=None):
        return self.alpha * tf.identity(inputs) + (1 - self.alpha) * self.activation_func(inputs)
