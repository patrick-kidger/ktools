import tensorflow.keras.initializers as init
import tools


class WithTrainable:
    """A context in which every layer in :layers: has their trainable flag set to :trainable:."""

    def __init__(self, layers, trainable=False):
        self.layers = list(layers)
        self.trainable = trainable
        self.original_trainables = []

    def __enter__(self):
        for layer in self.layers:
            self.original_trainables.append(layer.trainable)
            layer.trainable = self.trainable

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer, original_trainable in zip(self.layers, self.original_trainables):
            layer.trainable = original_trainable


uniq_name = tools.UniqueString(format_string='{string}__{index}')


class NearIdentity(init.Initializer):
    identity = init.Identity()

    def __init__(self, noise=init.truncated_normal(stddev=0.001)):
        self.noise = init.get(noise)

    def __call__(self, shape, dtype=None, partition_info=None):
        i = self.identity(shape, dtype, partition_info)
        n = self.noise(shape, dtype, partition_info)
        return i + n

    def get_config(self):
        return {'noise': self.noise.get_config()}


# So that keras.initializers.get works.
# https://github.com/keras-team/keras/issues/3867
init.get.__globals__['NearIdentity'] = NearIdentity
# What an awful hack.
