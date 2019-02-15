import os


def get(dataset_dir, dataset_name):
    """Gets the directory to find the requested dataset in. If :dataset_dir: is not None then it will be the directory
    used. If :dataset_dir: is None then the specified :dataset_name: will be looked up as an attribute in the
    dataset_dirs module, and its valued returned.
    """

    if dataset_dir is None:
        try:
            from . import dataset_dirs
        except ImportError:
            try:
                import dataset_dirs
            except ImportError as e:
                raise ImportError(f'Please create a "dataset_dirs" module somewhere on the PYTHONPATH, or inside the '
                                  f'ktools.datasets folder, listing where to find the dataset_dir "{dataset_name}". '
                                  f'This module should have an attribute called "{dataset_name}", whose value should '
                                  f'be a string specifying the folder where the dataset_dir is located.') from e

        dataset_dir = getattr(dataset_dirs, dataset_name)
    return os.path.expanduser(dataset_dir)


def show(x, y):
    """Calls imshow on x with label y."""

    # lazy import
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    ax.set_title(f'Label: {y}')
    fig.show()


def details(dataset_locals):
    """Parses the local dictionary of the dataset to determine certain details."""

    label_shape = dataset_locals['label_shape']
    if label_shape == ():
        last_layer_size = 1
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        assert len(label_shape) == 1
        last_layer_size = label_shape[0]
        if last_layer_size > 1:
            loss = 'categorical_crossentropy'
            last_layer_activation = 'softmax'
        else:
            loss = 'binary_crossentropy'
            last_layer_activation = 'sigmoid'
    return loss, last_layer_size, last_layer_activation
