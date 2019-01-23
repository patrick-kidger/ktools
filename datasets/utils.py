import os


def get(dataset_dir, dataset_name):
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
    # lazy import
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    ax.set_title(f'Label: {y}')
    fig.show()
