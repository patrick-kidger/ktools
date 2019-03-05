import numpy as np
import tensorflow as tf
import tensorflow.keras.initializers as init


class NearIdentity(init.Initializer):
    """An initializer which uses the identity plus some noise; the noise should be given by another initializer."""

    identity = init.Identity()

    def __init__(self, noise=init.truncated_normal(stddev=0.001), reshape=False):
        """
        Arguments:
            noise: The noise to add to the identity initializer; should be another intializer.
            reshape: Whilst identity intialization only makes sense for 2D input shapes, it can be extended to higher
                dimensions provided that the sizes of the higher dimensions are all one, by interpreting e.g. a shape of
                (1, 3, 4) as shape (3, 4) and then wrapping in the extra dimension afterwards. If this parameter is
                True (it defaults to False), then this interpretation is allowed.
        """
        self.noise = init.get(noise)
        self.reshape = reshape

    def __call__(self, shape, dtype=None, partition_info=None):
        if self.reshape:
            if len(shape) < 2 or len([i for i in shape if i != 1]) > 2:
                raise ValueError('Shape must be at least two dimensional, and not specify more than two dimensions of '
                                 'size greater than one.')
            squeezed_shape = []
            for dim_size in shape[:-2]:
                if dim_size != 1:
                    squeezed_shape.append(dim_size)

            if len(squeezed_shape) == 0:
                squeezed_shape = shape[-2:]
            elif len(squeezed_shape) == 1:
                dim_size = shape[-2]
                if dim_size != 1:
                    squeezed_shape.append(dim_size)
                else:
                    squeezed_shape.append(shape[-1])
        else:
            squeezed_shape = shape
        i = self.identity(squeezed_shape, dtype, partition_info)
        n = self.noise(squeezed_shape, dtype, partition_info)
        result = i + n
        if self.reshape:
            result = tf.reshape(i + n, shape)
        return result

    def get_config(self):
        return {'noise': {'class_name': self.noise.__class__.__name__, 'config': self.noise.get_config()}}


# So that keras.initializers.get works.
# https://github.com/keras-team/keras/issues/3867
init.get.__globals__['NearIdentity'] = NearIdentity
# What an awful hack.
