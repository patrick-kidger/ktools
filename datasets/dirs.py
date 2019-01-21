def get(dataset):
    try:
        from . import dataset_dirs
    except ImportError:
        try:
            import dataset_dirs
        except ImportError as e:
            raise ImportError(f'Please create a "dataset_dirs" module somewhere on the PYTHONPATH, or inside the '
                              f'ktools.src.datasets folder, listing where to find the dataset "{dataset}". This module '
                              f'should have an attribute called "{dataset}", whose value should be a string specifying '
                              f'the folder where the dataset is located.') from e

    return getattr(dataset_dirs, dataset)
