import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import threading
import time
import tools
import webbrowser


def tb_view(model, logdir=None, cmd=None):
    """Visualises a :model: in TensorBoard. (That is, everything in the model's Graph, which may actually be much larger
    than the model itself.)

    TensorBoard should automatically open; it is inconsistent whether the browser will automatically come to the front
    though.

    Extra arguments:
        :logdir: is the directory to save the model to prior to opening it in TensorBoard. Defaults to a randomly-named
            temporary directory.
        :cmd: is any command to call before launching TensorBoard, for example to open a virtual environment. This can
            be arbitrary shell code.
    """

    if logdir is None:
        logdir = f'/tmp/{tools.uuid2()}'
    inp = model.input
    if isinstance(inp, (tuple, list)):
        inp = inp[0]
    graph = inp.graph
    tf.summary.FileWriter(logdir=logdir, graph=graph).flush()

    def run_tensorboard():
        if cmd:
            tools.shell(f'{cmd}; tensorboard --logdir {logdir}')
        else:
            tools.shell(f'tensorboard --logdir {logdir}')
    thread = threading.Thread(target=run_tensorboard)
    thread.start()
    time.sleep(2)  # todo: actually detect when tensorboard is ready and open then. But this is almost always right.
    webbrowser.open_new_tab('http://localhost:6006')
    thread.join()


def plot_fn(fn, domain=(-2, 2), num_points=401, tensorflow=False, vector=False):
    """Plots the given univariate :fn: on the given :domain: at :num_points: equally-spaced points."""

    x = np.linspace(domain[0], domain[1], num_points)
    with tf.Session(graph=tf.Graph()) if tensorflow else tools.WithNothing() as sess:
        if vector:
            y = fn(x)
        else:
            y = [fn(xi) for xi in x]
        if tensorflow:
            y = sess.run(y)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    fig.show()
    return fig, ax


# http://parneetk.github.io/blog/cnn-cifar10/
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc'])+1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
