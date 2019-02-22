import numpy as np
import tensorflow as tf
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


# idea taken from https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
def periodize(tensor_to_periodize, margin_size):
    """Periodize the given tensor; that is, give it a margin that is filled in with values taken from the opposite side
    of the tensor.

    This is useful, for example, when wanting to apply a periodic convolutional layer. If the kernel is 7x7, for
    example, then add a margin of size 3 (=(7-1)/2) to the tensor and use 'valid' padding in the convolutional layer.
    (Although one has to be careful to specify a size-0 margin on the batch dimension!)

    Arguments:
        tensor_to_periodize: The tensor to periodize.
        margin_size: Should be a tuple of nonnegative integers; the length of the tuple should be the length of the
            shape of tensor_to_periodize. Dimension i of tensor_to_periodize will get a margin of size margin_size[i],
            meaning that the overall size of the dimension will then become
            tensor_to_periodize.shape[i] + 2 * margin_size[i].
            Alternatively this argument may be an integer, in which case the same margin will be applied across all
            axes.
            Alternatively this argument by a tuple of 2-tuples, each element of which is a nonnegative integer. The
            first element in the 2-tuple will be the margin at the start of the axis, and the second element will be the
            margin at the end of the axis.

    Returns:
        The periodized tensor.

    Examples:
        x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        periodize(x, 1).eval()  # array([4., 1., 2., 3., 4., 1.], dtype=float32)

        x = tf.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float))
        periodize(x, (0, 1)).eval()  # array([[4., 1., 2., 3., 4., 1.],
                                     #       [8., 5., 6., 7., 8., 5.]], dtype=float32)
        periodize(x, (1, 0)).eval()  # array([[5., 6., 7., 8.],
                                     #        [1., 2., 3., 4.],
                                     #        [5., 6., 7., 8.],
                                     #        [1., 2., 3., 4.]], dtype=float32)

    """

    if tensor_to_periodize.dtype not in (tf.float64, tf.float32, tf.int32):
        # not tested other dtypes, not clear if they'd work
        raise ValueError(f'tensor_to_periodize must be of dtypes tf.float64, tf.float32 or tf.int32; given '
                         f'{tensor_to_periodize.dtype}')
    tensor_shape = tensor_to_periodize.shape.as_list()
    if isinstance(margin_size, int):
        # this will satisfy the next if block
        margin_size = (margin_size,) * len(tensor_shape)
    if isinstance(margin_size, (tuple, list)):
        new_margin_size = []
        for margin_size_i in margin_size:
            if isinstance(margin_size_i, int):
                new_margin_size.append((margin_size_i, margin_size_i))
            else:
                new_margin_size.append(margin_size_i)
        margin_size = tuple(new_margin_size)
    if len(tensor_shape) != len(margin_size):
        raise ValueError(f'tensor_to_periodize.shape must have the same length as margin_size. Given '
                         f'tensor_to_periodize.shape: {tensor_to_periodize.shape}, given margin_size: {margin_size}')
    for matrix_shape_i, margin_size_i in zip(tensor_shape, margin_size):
        # asserts the margin_size_i is a 2-tuple, with a nice error message if it isn't
        start_margin_size_i, end_margin_size_i = margin_size_i
        for entry_margin_size_i in (start_margin_size_i, end_margin_size_i):
            if not isinstance(entry_margin_size_i, int) or entry_margin_size_i < 0:
                raise ValueError(f'All entries in margin_size must be nonnegative integers.')

    for i, (matrix_shape_i, margin_size_i) in enumerate(zip(tensor_shape, margin_size)):
        start_margin_size_i, end_margin_size_i = margin_size_i
        if matrix_shape_i is None:
            # Can't statically periodize along unknown dimension sizes
            continue

        if start_margin_size_i == 0 and end_margin_size_i == 0:
            # Nothing would go wrong if we didn't have this here, but there's no point wasting time multiplying by an
            # identity matrix
            continue

        # make something akin to an identity matrix, e.g. for a tensor whose i-th dimension if of size i, for which we
        # want to add a margin of size 2, we create the matrix
        # 0 1 0   \
        # 0 0 1   /  adds margin of width 2
        # 1 0 0   \
        # 0 1 0    | keeps the original data
        # 0 0 1   /
        # 1 0 0   \
        # 0 1 0   /  adds margin of width 2
        to_roll = np.zeros(matrix_shape_i)
        one_index = (-start_margin_size_i) % matrix_shape_i
        to_roll[one_index] = 1.0
        new_length = matrix_shape_i + start_margin_size_i + end_margin_size_i
        rolled = np.array([np.roll(to_roll, roll_index) for roll_index in range(new_length)])

        rolled = tf.constant(rolled, dtype=tensor_to_periodize.dtype)
        tensor_to_periodize = tf.tensordot(rolled, tensor_to_periodize, (1, i))
        perm = list(range(1, i + 1)) + [0] + list(range(i + 1, len(tensor_shape)))
        tensor_to_periodize = tf.transpose(tensor_to_periodize, perm)

    return tensor_to_periodize
