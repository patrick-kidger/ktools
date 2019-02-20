import multiprocessing as mp
import numpy as np
import random
import tensorflow.keras as keras


class TransformedSequence(keras.utils.Sequence):
    """Creates a new keras.utils.Sequence that applies a transformation to a previous one."""

    def __init__(self, sequence, transform):
        self.sequence = sequence
        self.transform = transform

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.transform(self.sequence[item])


# As a general rule, it seems like the capabilities of Tensorflow and Keras' for taking python-generated data from
# generators isn't too great.
# - keras.Model.fit_generator(..., with_multiprocessing=True, workers=32) doesn't seem to correctly set a random seed
#   for each worker process, so each process will generate the same data.
# - tf.data.Dataset doesn't offer any multiprocessing-based parallelism in its from_generator method. There are hacks
#   around this, like using a parallel map applied to a dataset drawing null data from a generator, but that's just
#   ugly. More than that, it incurs an overhead, because the map has to be wrapped in a tf.py_func.
# So it falls to us to write our own multiprocessing, which can then fed into a non-multiprocessed
# keras.Model.fit_generator, for example.

class MultiprocessGenerator:
    """Runs multiple copies of the given generator in separate processes (with different random seeds!), and puts their
    results in a queue. Iterating through this object will then get items from the queue. Useful for parallelising
    expensive generators, which are needed to generate large amounts of data.

    Once the generator has been created it must be .start() ed. After it is no longer desired, call .terminate() to
    tidy up the processes its spawned to do its work. The instance may be used as a context to handle such things
    automatically.

    Example:

        def f():
            while True:
                yield 4

        m = MultiprocessGenerator(f())
        with m:
            next(m)  # outputs 4
            next(m)  # outputs 4
        # don't use m again after this, as its been terminated.
    """

    def __init__(self, generator, workers=8, max_queue_size=10):
        """Runs multiple copies of the given :generator: in :workers: number of separate processes."""

        def gen(i, num_workers):
            def gen_():
                # So each process often has to know its identity, and how many other workers there are, for example to
                # avoid duplicating data. How best to let them know?
                # We could call some method of the generator: generator.set_id(...), but then not only does every
                # generator we want to use have to support this method, but _also_ every generator which wraps other
                # generators does as well. For example batch_generator, below. And we don't really want to have to
                # mandate that every generator we care to use should supply this method, or be wrapped in a class
                # supplying this method.
                # So the alternative is to store this information as global state somewhere. (Yes yes, yuck. But the
                # identity of a process is a fundamentally global-level piece of information. If anything actually it is
                # slightly more than global, as it is invariant across an entire kernel, not just a module.) The obvious
                # place to put this would be as an attribute on the multiprocessing module, but the multiprocessing
                # module offers several 'contexts' in which it can work, each of which is essentially a fake of the
                # true multiprocessing module, just copying its interface. So it's actually nontrivial (although still
                # not terribly hard) to get access to the 'same' multiprocessing module everywhere. So the next most
                # obvious place to put the attribute is on the sys module, which is what we do.
                import sys
                if hasattr(sys, '_WORKER_ID'):
                    raise RuntimeError('sys module already has _WORKER_ID attribute')
                if hasattr(sys, '_WORKER_COUNT'):
                    raise RuntimeError('sys module already has _WORKER_COUNT attribute')
                sys._WORKER_ID = i
                sys._WORKER_COUNT = num_workers
                random.seed(i)
                np.random.seed(i)
                while True:
                    self._queue.put(next(generator))
            return gen_

        self._queue = mp.Queue(maxsize=max_queue_size)
        self._processes = [mp.Process(target=gen(i, workers)) for i in range(workers)]

        self._started = False
        self._terminated = False

    def start(self):
        """Starts the processes which populate this generator."""

        self._started = True
        for process in self._processes:
            process.daemon = True
            process.start()
        return self  # for chaining

    def __del__(self):
        self.terminate()

    def terminate(self):
        """Terminates the processes which populate this generator."""

        if self._started and not self._terminated:
            for process in self._processes:
                process.terminate()
                process.join()
            self._queue.close()
            self._terminated = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self._started:
            raise RuntimeError(f'{self.__class__} has not yet been started; cannot iterate on it.')
        if self._terminated:
            raise RuntimeError(f'{self.__class__} has already been terminated; cannot iterate on it.')
        return self._queue.get(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


def _map_structures(objs, modify=lambda objs: [obj_i.shape for obj_i in objs]):
    """Walks through several structures, all of which are assumed to have the same structure as each other. All of them
    are assumed not to have any self-referential links. Eventually it expects to find a 'primitive' object, namely a
    numpy.ndarray, int, float, or bool - in each structure. All items at the same location in each structure will then
    be put together in a list, and have :modify: called on them. (int, float, bool will be cast to numpy arrays first.)

    The result will be a structure of the same sort as the structures given, but with the result of the call to modify
    in each location in the structure.

    The default :modify: specifies the shape of the elements of the structures. Another sensible choice is np.stack,
    to stack numpy arrays together.

    It is assumed that the non-'primitive' elements of the structure consists only of tuples, lists, dictionaries, and
    compositions thereof.
    """

    obj_1 = objs[0]
    if isinstance(obj_1, np.ndarray):
        return modify(objs)
    elif isinstance(obj_1, list):
        # it's actually important to keep the list/tuple distinction
        # https://github.com/keras-team/keras/issues/2568
        return [_map_structures(obj_i, modify=modify) for obj_i in zip(*objs)]
    elif isinstance(obj_1, tuple):
        return tuple(_map_structures(obj_i, modify=modify) for obj_i in zip(*objs))
    elif isinstance(obj_1, (int, float, bool)):
        return modify([np.array(obj_i) for obj_i in objs])
    elif isinstance(obj_1, dict):
        returndict = {}
        for key in obj_1:
            vals = [obj_i[key] for obj_i in objs]
            returndict[key] = _map_structures(vals, modify=modify)
        return returndict
    else:
        raise ValueError(f'_unpack_structure does not understand the type of the object {str(obj_1)[:50]}, which is of '
                         f'type {type(obj_1)}')


# So in light of the comment above about multiprocessing, why are we doing our own batching as well? Well, the only
# alternative seems to be feeding the results of MultiprocessGenerator into tf.data.Dataset.from_generator, and then
# # calling its batch method. Now this does sound like a reasonable option. But even besides how painful it is to
# specify the types/shapes to its liking (I never did figure out how to make it handle a feature that's a list; I had to
# use dictionaries for multiple features), it's also super slow. When generating a small dataset
# (of full size (100 000, 2)), then I found my solution took 8s whilst tf.data.Dataset took 36s.

def batch_generator(generator, batch_size=128):
    """Takes a :generator: generating individual samples and batches them up into batches of size :batch_size:."""

    while True:
        features, labels = zip(*[next(generator) for _ in range(batch_size)])
        batch_features = _map_structures(features, modify=np.stack)
        labels_features = _map_structures(labels, modify=np.stack)
        yield batch_features, labels_features