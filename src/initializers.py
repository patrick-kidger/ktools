import tensorflow.keras.initializers as init


class NearIdentity(init.Initializer):
    """An initializer which uses the identity plus some noise; the noise should be given by another initializer."""

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
