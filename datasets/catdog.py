import functools as ft
import numpy as np
import os
import tensorflow.keras.preprocessing.image as image
import tools

from . import utils


@ft.lru_cache(maxsize=1)
def _data(base_dir=None):
    base_dir = utils.get(base_dir, 'catdog')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'test')
    subdirs = ['cat', 'dog']

    def count_files(dir_):
        count = 0
        for subdir in subdirs:
            subdir = os.path.join(dir_, subdir)
            for name in os.listdir(subdir):
                if os.path.isfile(os.path.join(subdir, name)):
                    count += 1
        return count

    train_filecount = count_files(train_dir)
    validation_filecount = count_files(validation_dir)
    assert train_filecount > 0
    assert validation_filecount > 0
    return train_dir, validation_dir, subdirs, train_filecount, validation_filecount


def datagens(batch_size=128, train_datagen=None, validation_datagen=None, base_dir=None):
    """Creates some generators for the data, along with some auxiliary information."""

    if train_datagen is None:
        train_datagen = image.ImageDataGenerator(rotation_range=40,
                                                 width_shift_range=0.3,
                                                 height_shift_range=0.3,
                                                 shear_range=0.2,
                                                 zoom_range=0.3,
                                                 fill_mode='nearest',
                                                 horizontal_flip=True,
                                                 preprocessing_function=lambda x: -1 + x / 127.5)  # scale to [-1, 1]
    if validation_datagen is None:
        validation_datagen = image.ImageDataGenerator(preprocessing_function=lambda x: -1 + x / 127.5)  # scale to [-1, 1]

    train_dir, validation_dir, subdirs, train_filecount, validation_filecount = _data(base_dir)

    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(150, 150),
                                                        batch_size=batch_size, class_mode='binary', classes=subdirs)
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(150, 150),
                                                                  batch_size=1, class_mode='binary',
                                                                  classes=subdirs)
    train_steps = np.ceil(train_filecount / batch_size)
    return tools.Record(train_generator=train_generator, validation_generator=validation_generator,
                        train_steps=train_steps, validation_steps=validation_filecount,
                        train_datasize=train_filecount, validation_datasize=validation_filecount)


def show(batch_size=1, train_datagen=None, validation_datagen=None):
    d = datagens(batch_size=batch_size, train_datagen=train_datagen, validation_datagen=validation_datagen)
    x, y = next(d.train_generator)
    x = x[0]  # unpack from batch dimension
    y = y[0]  #
    x = np.round((x + 1) * 127.5).astype(int)  # undo the preprocessing scaling
    if y == 0:
        y = 'Cat'
    else:
        y = 'Dog'
    utils.show(x, y)


feature_shape = (150, 150, 3)
label_shape = ()
