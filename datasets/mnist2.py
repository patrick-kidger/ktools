import numpy as np
import tensorflow.keras.datasets as datasets
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.utils as utils
import tools


def datagens(batch_size=32, train_datagen=None, validation_datagen=None):
    if train_datagen is None:
        train_datagen = image.ImageDataGenerator(rotation_range=30,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 shear_range=0.2,
                                                 zoom_range=0.2,
                                                 fill_mode='nearest',
                                                 data_format='channels_first')
    if validation_datagen is None:
        validation_datagen = image.ImageDataGenerator(data_format='channels_first')

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    def _preprocess_x(x):
        return (x.reshape(x.shape[0], 1, 28, 28).astype('float32') / 127.5) - 1    # scale to [-1, 1]
    x_train = _preprocess_x(x_train)
    y_train = utils.to_categorical(y_train)
    x_test = _preprocess_x(x_test)
    y_test = utils.to_categorical(y_test)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = validation_datagen.flow(x_test, y_test, batch_size=1)

    return tools.Record(train_generator=train_generator, validation_generator=validation_generator,
                        train_steps=np.ceil(len(x_train) / batch_size), validation_steps=len(x_test))
