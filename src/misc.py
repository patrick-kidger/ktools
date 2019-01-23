import tensorflow.keras as keras
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


class TransformedSequence(keras.utils.Sequence):
    """Creates a new keras.utils.Sequence that applies a transformation to a previous one."""

    def __init__(self, sequence, transform):
        self.sequence = sequence
        self.transform = transform

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.transform(self.sequence[item])


uniq_name = tools.UniqueString()
