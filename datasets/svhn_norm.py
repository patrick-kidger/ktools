import functools as ft
import numpy as np
import os
import scipy.io as sio
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.utils as keras_utils
import tools

from . import utils


@ft.lru_cache(maxsize=1)
def _data(base_dir=None):
    base_dir = utils.get(base_dir, 'svhn_norm')
    train_data = sio.loadmat(os.path.join(base_dir, 'train_32x32.mat'))
    extra_data = sio.loadmat(os.path.join(base_dir, 'extra_32x32.mat'))
    test_data = sio.loadmat(os.path.join(base_dir, 'test_32x32.mat'))

    x_train_1 = np.moveaxis(train_data['X'], 3, 0)  # put the batch axis first. Channels axis is now last.
    y_train_1 = train_data['y'].reshape(-1)  # by default is shape (batch, 1); this just unpacks to shape (batch,).
    x_train_2 = np.moveaxis(extra_data['X'], 3, 0)
    y_train_2 = extra_data['y'].reshape(-1)
    x_train = np.concatenate((x_train_1, x_train_2), axis=0)
    y_train = np.concatenate((y_train_1, y_train_2), axis=0)
    y_train = np.array([0 if y == 10 else y for y in y_train])
    y_train = keras_utils.to_categorical(y_train)
    x_test = np.moveaxis(test_data['X'], 3, 0)
    y_test = test_data['y'].reshape(-1)
    y_test = np.array([0 if y == 10 else y for y in y_test])
    y_test = keras_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def datagens(batch_size=128, train_datagen=None, validation_datagen=None, base_dir=None):
    x_train, y_train, x_test, y_test = _data(base_dir)

    if train_datagen is None:
        train_datagen = image.ImageDataGenerator(rotation_range=30,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 shear_range=0.2,
                                                 zoom_range=0.2,
                                                 fill_mode='nearest',
                                                 # normalise values to [-1, 1]
                                                 preprocessing_function=lambda x: -1 + x / 127.5,
                                                 data_format='channels_last')
    if validation_datagen is None:
        validation_datagen = image.ImageDataGenerator(preprocessing_function=lambda x: -1 + x / 127.5,
                                                      data_format='channels_last')

    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = validation_datagen.flow(x_test, y_test, batch_size=1)

    return tools.Record(train_generator=train_generator, validation_generator=validation_generator,
                        train_steps=np.ceil(len(x_train) / batch_size), validation_steps=len(x_test),
                        train_datasize=len(x_train), validation_datasize=len(x_test))


def show(data=None, datagen=None):
    if data is None:
        d = datagens(batch_size=1, train_datagen=datagen)
        x, y = next(d.train_generator)
    else:
        x, y = data
    x = x[0]  # unpack from batch dimension
    y = y[0]  #
    x = np.round((x + 1) * 127.5).astype(int)  # undo the preprocessing scaling
    y = list(y).index(1)  # undo the effect of to_categorical
    utils.show(x, y)


feature_shape = (32, 32, 3)
label_shape = (10,)

loss, last_layer_size, last_layer_activation = utils.details(locals())
