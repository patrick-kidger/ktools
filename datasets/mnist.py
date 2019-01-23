import numpy as np
import tensorflow.keras.datasets as datasets
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.utils as keras_utils
import tools

from . import utils


def datagens(batch_size=128, train_datagen=None, validation_datagen=None):
    if train_datagen is None:
        train_datagen = image.ImageDataGenerator(rotation_range=30,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 shear_range=0.2,
                                                 zoom_range=0.1,
                                                 fill_mode='nearest',
                                                 data_format='channels_first')
    if validation_datagen is None:
        validation_datagen = image.ImageDataGenerator(data_format='channels_first')

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    def _preprocess_x(x):
        return (x.reshape(x.shape[0], 1, 28, 28).astype('float32') / 127.5) - 1    # scale to [-1, 1]
    x_train = _preprocess_x(x_train)
    y_train = keras_utils.to_categorical(y_train)
    x_test = _preprocess_x(x_test)
    y_test = keras_utils.to_categorical(y_test)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = validation_datagen.flow(x_test, y_test, batch_size=1)

    return tools.Record(train_generator=train_generator, validation_generator=validation_generator,
                        train_steps=np.ceil(len(x_train) / batch_size), validation_steps=len(x_test),
                        train_datasize=len(x_train), validation_datasize=len(x_test))


def show(batch_size=1, train_datagen=None, validation_datagen=None):
    d = datagens(batch_size=batch_size, train_datagen=train_datagen, validation_datagen=validation_datagen)
    x, y = next(d.train_generator)
    x = x[0][0]  # unpack from batch and channel dimensions, which should both be of size 1.
    y = y[0]  # unpack from batch dimension
    y = list(y).index(1)  # undo the effect of to_categorical
    utils.show(x, y)


feature_shape = (1, 28, 28)
label_shape = (10,)
