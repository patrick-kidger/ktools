import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as keras_layers
import tools

from . import misc
from . import scopes


_sentinel = object()  # Used to detect cycles in webs


class Knot:
    """Represents the vertex of a graph. Here we instead call them Knots in webs, however, to avoid confusion with the
    usual TensorFlow meaning of 'Graph'.

    Must have a layer attached to it, and it may be connected to other knots, which will be treated as its inputs.
    Calling make() on a Knot will then call its layer on the layers of inputs Knots (recursively).

    Whilst Keras does have Nodes which keeps track of something similar, they don't have quite the flexibility that
    we're after to make this work.

    The idea is that this is a wrapper around Keras layers, which may be called in exactly the same manner as Keras'
    functional API. Applying these calls to the Keras layers is handed in the Knot.make method; this allows for
    substituting some layers for other layers and then rebuilding easily. In particular the Knot.model method converts
    the web of Knots into a Keras Model.

    Note that all errors from trying to assemble the layers - e.g. incorrect shapes or cyclic webs - will only be
    noticed when actually assembling, via either Knot.make or Knot.model.

    Example:
        input = Knot(keras.layers.Input(shape=(2,)))
        x = Knot(keras.layers.Dense(1024))(input)  # calling returns the Knot instance wrapping the Dense layer.
        y = Knot(keras.layers.Dense(512))(input)
        c = Knot(keras.layers.Concatenate())([x, y])
        z = Knot(keras.layers.Dense(1024))(c)
        keras_model = z.model()
        ...
        # Now we want to replace the dense layer of size 512 with something bigger:
        y.set_layer(keras.layers.Dense(1024))
        keras_model_2 = z.model()

    Example:
        input = Knot(keras.layers.Input(shape=(2,)))
        x = Knot(keras.layers.Dense(1024))(input)
        y = Knot(keras.layers.Dense(1024))(x)
        z = Knot(keras.layers.Dense(1024))(y)
        tensorflow_tensor = z.make()
        ...
        # Now we want to replace the Knot y with some other Knot entirely. (In this simple example this is essentially
        # equivalent to just doing y.set_layer(keras.layers.Dense(2048)), as above.)
        w = Knot(keras.layers.Dense(2048))
        y.replace(w)  # y is no longer attached to anything; x and z now connect to w instead.
        tensorflow_tensor_2 = z.make()
        ...
        # Now we want to replace the Knot w with a web of two Knots, a and b:
        a = Knot(keras.layers.Dense(512))
        b = Knot(keras.layers.Dense(512))(a)
        w.replace_multi(input_knots=a, output_knots=b)  # w is no longer attached to anything
        tensorflow_tensor_3 = z.make()
        # essentially performing what z.model() would do for us:
        keras_model = keras.Model(inputs=input.layer, outputs=tensorflow_tensor_3)
    """

    def __init__(self, layer, data=None):
        """
        See also Knot.__doc__.

        Arguments:
            :layer: Should be a callable (e.g. a keras Layer) associated with this knot. This should take a tensor as an
                input and return a tensor as an output. (Or lists of tensors.)
            :data: Will be stored as a 'data' attribute on this node; nothing else is done with it. Useful for storing
                custom information about the node. Defaults to None.
        """
        self._layer = layer
        self.data = data

        self._input_knots = None
        self._output_knots = set()  # Those knots which depend on us, i.e. have us as or in their _input_knots.
        self.scope_names = None

    def clone(self, inputs=False, layers=False, weights=False):
        """Creates a clone of the Knot. This means that it will share the same layer, data, and scope names. If :inputs:
        is True then it will also share the same input Knots. If :layers: is True then the layer will also be cloned
        (without copying weights). If :weights: is also True then weights will also be copied.
        """

        if layers:
            if self.input_knots is None:
                shape = self.layer.shape[1:]  # remove batch dimension
                dtype = self.layer.dtype
                if isinstance(self.layer, tf.SparseTensor):
                    sparse = True
                    name = 'input'    # Sparse Tensors don't have names, oddly
                else:
                    sparse = False
                    name = self.layer.name.partition(':')[0]
                layer_ = keras_layers.Input(shape=shape, dtype=dtype, name=name, sparse=sparse)
            else:
                config = self.layer.get_config()
                config['name'] = misc.uniq_name(config['name'])
                if weights:
                    config['weights'] = self.layer.get_weights()
                layer_ = self.layer.__class__.from_config(config)
        else:
            layer_ = self.layer

        cloned_self = self.__class__(layer_, data=self.data)
        if inputs:
            cloned_self(self.input_knots)
        cloned_self.set_scope_names(self.scope_names)
        return cloned_self

    # @staticmethod
    # def _original_name(name):
    #     """Takes a name for a Layer or Tensor and returns a name of the same type which can be used to create a new
    #     Layer or Tensor.
    #
    #     For example if :name: is 'dense_3' then this will return 'dense'.
    #     """
    #
    #     try:
    #         return name.original_string
    #     except AttributeError:  # If the name is just a string passed to the Input, or not specified.
    #         name = name.partition(':')[0]  # Get rid of the ':0' suffix if it's present
    #
    #         # Strip the disambiguation if it's present
    #         # This isn't totally foolproof. Nothing stops one from passing 'hello_40' as a name; this is the
    #         # same name as you'd get if you'd created 41 tensors called 'hello'.
    #         pieces = name.rpartition('_')
    #         if pieces[0] != '' and pieces[2].isdigit():  # disambiguation present
    #             name = pieces[0]
    #         return name

    @property
    def input_knots_list(self):
        """All of the input knots as a list which may be iterated over."""

        if self._input_knots is None:
            return []
        elif isinstance(self._input_knots, (tuple, list)):
            return self._input_knots
        else:
            return [self._input_knots]

    @property
    def input_knots(self):
        """All of the input knots. There are three expected formats: None (corresponding to no inputs Knots), an
        instance of Knot, or a list or tuple or Knots.
        """

        return self._input_knots

    @property
    def output_knots(self):
        return self._output_knots

    @property
    def layer(self):
        """The callable (e.g. a Keras layer) associated with this Knot."""

        return self._layer

    @property
    def layer_name(self):
        if isinstance(self.layer, tf.Tensor):
            return self.layer.name.rsplit(':', 1)[0].rsplit('/', 1)[-1]
        else:
            return self.layer.name

    def set_layer(self, layer):  # somehow this seems nicer than using property.setter.
        """Sets the callable (e.g. a Keras layer) associated with this Knot."""

        self._layer = layer

    def set_scope_names(self, scope_names):
        """Sets the absolute scope that the layers will be called in to that of :scope_names:.

        Arguments:
            scope_names: A string specifying the scope(s). e.g. 'Wrapper1' or 'Wrapper1/Wrapper2'.
        """

        if scope_names and scope_names[-1] != '/':
            scope_names += '/'
        self.scope_names = scope_names

    def register_current_scopes(self):
        """Sets the absolute scope that the layers will be called in to whatever the current scope is."""
        self.set_scope_names(scopes.get_current_scopes())

    def prepend_current_scopes(self):
        """Sets the absolute scope that the layers will be called in to whatever the current scope is plus whatever the
        Knot currently has stored as the scope."""
        pieces = [scopes.get_current_scopes()]
        if self.scope_names:  # in particular not '' or None
            pieces.append(self.scope_names)
        self.set_scope_names('/'.join(pieces))

    def _add_output_knot(self, knot):
        self._output_knots.add(knot)

    def _remove_output_knot(self, knot):
        self._output_knots.remove(knot)

    def __call__(self, input_, register_scopes=True):
        """Call the Knot as you would a Keras layer; see Knot.__doc__.

        Arguments:
            input_: another Knot that this one will be called on.

        Returns:
            self, primarily so that the functional API is straightforward.
        """

        if isinstance(input_, (tuple, list)):
            assert len(input_) > 0
            for inp in input_:
                assert isinstance(inp, Knot)
        else:
            assert isinstance(input_, Knot)

        if register_scopes:
            self.register_current_scopes()

        for inp in self.input_knots_list:
            # noinspection PyProtectedMember
            inp._remove_output_knot(self)

        self._input_knots = input_
        for inp in self.input_knots_list:
            # noinspection PyProtectedMember
            inp._add_output_knot(self)

        return self  # for chaining

    def replace(self, knot):
        """Replaces this Knot's spot in the web with the given :knot:.

        Arguments:
            knot: another Knot to replace this one with.
        """
        self.replace_multi([knot], knot)

    # todo: document the fact that we have to clone self if its used in between last knot and first knots
    def replace_multi(self, first_knots, last_knot):
        """Replaces this Knot's spot in the web with multiple other Knots.

        The idea is usually that the given input knots and output Knot will be connected to each other - indeed, may
        even be the same Knot - and so serve to expand the web. This is not enforced however, so this may separate the
        web if desired.

        Arguments:
            first_knots: A Knot or list or tuple of Knots, which will take on the inputs of the existing Knot.
            last_knot: A Knot which will replace this Knot in all outputs of the existing Knot.
        """

        if not isinstance(first_knots, (tuple, list)):
            first_knots = [first_knots]

        # have our current input knots forget about us
        for inp in self.input_knots_list:
            # noinspection PyProtectedMember
            inp._remove_output_knot(self)

        # have the input knots inherit our input, and have the input depend on it.
        if self.input_knots:
            for knot in first_knots:
                knot(self.input_knots, register_scopes=False)

        # update our outputs to depend on the new output knot instead
        for n in self.output_knots:
            if isinstance(n._input_knots, (tuple, list)):
                n._input_knots = [last_knot if x is self else x for x in n._input_knots]
            else:
                n._input_knots = last_knot

        # let the last knot know that its output knots are our output knots
        for output_knot in self.output_knots:
            last_knot._add_output_knot(output_knot)

        # remove our own connections
        self._input_knots = None
        self._output_knots = set()

    def map_web(self, fn, memodict=None):
        """Applies the given function to every Knot in the web which are accessible as inputs (of inputs of inputs
        of...) this one, and stores the result of this function in the given memorisation dictionary.

        That is, the function is called on all of Knots which are 'earlier' in the web: treating the web as a directed
        graph, then all Knots which may be reached by going backwards along edges will have the function applied to
        them. The function will be called precisely once on each Knot, even if the Knot is reachable in multiple
        different ways. (Unless the given function does something to mess with that behaviour, of course, but that would
        be unusual.)

        The function will be called on the 'earliest' Knots first.

        Arguments:
            fn: The function to call on each Knot. It should take three arguments. The first is the Knot that the
                function is called on. The second is a convenience: the result of the function on all Knots which are
                inputs to the Knot. The third is the memorisation dictionary which stores the results of the function
                calls on each Knot, with the Knots as keys and the result of the function on that Knot as values. (Hence
                why the second argument is a convenience, as these may also be extracted from this dictionary.)
            memodict: The dictionary in which the results are recorded. Passing it in manually allows access to it after
                the call to map_web is completed, or to reuse the same memodict between multiple calls to map_web.

        Returns:
             The result of calling the given function :fn: on this Knot.
        """

        if memodict is None:
            memodict = {}

        try:
            knot_result = memodict[self]
        except KeyError:
            memodict[self] = _sentinel
            if isinstance(self.input_knots, (tuple, list)):
                mapped_inputs = [i.map_web(fn, memodict) for i in self.input_knots]
            elif self.input_knots is None:
                mapped_inputs = None
            else:
                mapped_inputs = self.input_knots.map_web(fn, memodict)
            knot_result = fn(self, mapped_inputs, memodict)
            memodict[self] = knot_result
        else:
            if knot_result is _sentinel:
                raise CycleException

        return knot_result

    def make(self, memodict=None):
        """Makes the TensorFlow Tensor from this Knot and all Knots 'earlier' than it (in the sense of
        Knot.map_web.__doc__).

        The Tensor returned is equivalent to the usual Tensor created from the Keras functional API. That is, the
        following code snippets generate equivalent tensors:

        Example:
            i = keras.layers.Input(shape=(2,))
            d = keras.layers.Dense(3)(i)
            d = keras.layers.Dense(3)(d)
            d = keras.layers.Dense(3)(d)
            tensor = d
            ...
            i = Knot(keras.layers.Input(shape=(2,)))
            d = Knot(keras.layers.Dense(3))(i)
            d = Knot(keras.layers.Dense(3))(d)
            d = Knot(keras.layers.Dense(3))(d)
            tensor = d.make()

        Arguments:
            memodict: The memorisation dictionary, in the sense of Knot.map_web.__doc__, for storing the tensors
                produced from intermediate layers.

        Returns:
            The tensor from calling the Keras layers.
        """

        def make_fn(knot, made_inputs, memodict_):
            if made_inputs is None:
                assert isinstance(knot.layer, tf.Tensor)
                input_layers = memodict_.setdefault('__inputs', set())
                input_layers.add(knot.layer)
                return knot.layer  # It's an input knot
            with tf.name_scope(knot.scope_names):
                return knot.layer(made_inputs)

        return self.map_web(make_fn, memodict)

    def model(self, input_knots=None):
        """See model_web.__doc__.

        Constructs a Keras Model from this Knot and all Knots 'earlier' than it. This method just calls model_web with
        this Knot the output Knot.
        """

        return model_web(self, input_knots)

    def __deepcopy__(self, memodict=None, layers=False, weights=False):
        return deepcopy_web(self, memodict, layers=layers, weights=weights)

    def deepcopy(self, memodict=None, layers=False, weights=False):
        """See deepcopy_web.__doc__.

        Makes a deepcopy of this Knot and all Knots 'earlier' than it.
        """

        return self.__deepcopy__(memodict, layers=layers, weights=weights)

    def to_networkx(self):
        """Converts this Knot and all 'earlier' knots to a NetworkX graph; see networkx_web.__doc__ for more
        information."""

        # noinspection PyTypeChecker
        return to_networkx_web(self)

    def draw(self, mode='pydot-300', returnval=False):
        """Display a visual of this web. See draw_web.__doc__ for more information."""

        # noinspection PyTypeChecker
        return draw_web(self, mode=mode, returnval=returnval)


class CycleException(tools.DefaultException):
    default_msg = 'Cycle detected in web.'


def _make_web(output_knots, input_knots=None):
    if input_knots is not None:
        if not isinstance(input_knots, (tuple, list)):
            input_knots = [input_knots]
        for input_knot in input_knots:
            assert not input_knot.input_knots_list  # Inputs can't have inputs!
    if isinstance(output_knots, (tuple, list)):
        as_list = True
    else:
        output_knots = [output_knots]
        as_list = False

    # Use the same memodict for all of them, as else each output tensor will exist in a disjoint subgraph to the others
    memodict = {}
    outputs = [output_knot.make(memodict=memodict) for output_knot in output_knots]
    inputs = memodict['__inputs']
    if input_knots is not None:
        assert inputs == {input_knot.layer for input_knot in input_knots}

    if as_list:
        return inputs, outputs
    else:
        return inputs, outputs[0]


def make_web(output_knots, input_knots=None):
    """Makes Tensorflow Tensors from a collection of Knots and all Knots 'earlier' than them (in the sense of
    Knot.map_web.__doc__).

    See also Knot.make.__doc__.

    Arguments:
        output_knots: A Knot or list or tuple of Knots, whose layers form the outputs of the Model.
        input_knots: May optionally specify the input Knots to the web. This does not affect the construction of the
            Model, as the input Knots to the web are already specified from the construction of the web, but instead
            just serves to to validate that the input Knots are indeed the expected ones.

    Returns:
        The constructed TensorFlow Tensors.
    """

    inputs, outputs = _make_web(output_knots, input_knots)
    return outputs


def model_web(output_knots, input_knots=None):
    """Constructs a Keras Model from a collection of Knots and all Knots 'earlier' than them (in the sense of
    Knot.map_web.__doc__).

    Arguments:
        output_knots: A Knot or list or tuple of Knots, whose layers form the outputs of the Model.
        input_knots: May optionally specify the input Knots to the web. This does not affect the construction of the
            Model, as the input Knots to the web are already specified from the construction of the web, but instead
            just serves to to validate that the input Knots are indeed the expected ones.

    Returns:
        The constructed Keras Model.
    """

    inputs, outputs = _make_web(output_knots, input_knots)
    inputs = list(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def deepcopy_web(knots, memodict=None, layers=False, weights=False):
    """Creates a deepcopy of the web.

    Copies the specified Knots and all Knots 'earlier' in the web, in the sense of Knot.map_web.__doc__, and returns the
    copies of those Knots. The Knots themselves will be copies, whilst the other information - layer, data and scopes -
    will be the same objects as in the original web.

    Arguments:
        knots: A Knot or tuple or list of Knots, specifying the Knots to be copied (which will then result in copying
            all Knots 'earlier' than them as well.)
        memodict: The memorisation dictionary, in the sense of Knot.map_web.__doc__. Passing a known dictionary
        allows for accessing the key: value pairs corresponding to what Knot has been copied to create what Knot.
        layers: Whether to copy the layers or use the same layer. Defaults to False.
        weights: If copying layers, whether to also copy their weights, or let the weights in the new layer initialize
            as before. Defaults to False, i.e. no copying.

    Returns:
        The copies of :knots:, either as a single Knot or as a list of Knots, depending on the input format.
    """

    def deepcopy_fn(knot, copied_inputs, _):
        # don't copy the layers and such
        new_knot = knot.clone(layers=layers, weights=weights)
        # do deepcopy our input knots
        if copied_inputs is not None:
            with tf.name_scope(knot.scope_names):
                new_knot(copied_inputs)
        return new_knot

    if isinstance(knots, (tuple, list)):
        return [knot.map_web(deepcopy_fn, memodict) for knot in knots]
    else:
        return knots.map_web(deepcopy_fn, memodict)


def to_networkx_web(knots):
    """Converts some Knots into a NetworkX DiGraph. The resulting DiGraph will have the Knots as its nodes.

    Note that there are good reasons not to just use NetworkX DiGraphs in the first place: NetworkX is based around
    using a Graph as the central object, for which its nodes may be anything. Here, however, we use Knots as the central
    object, which are assembled into a web. (Note the capitalisation in each case: the nodes of a Graph are not
    themselves NetworkX objects; nor is a web of Knots an object itself.)

    Arguments:
        knots: A Knot or tuple or list of Knots, specifing the 'outputs' of the web. All 'earlier' Knots, in the sense
            of Knot.map_web.__doc__, will feature as part of the result DiGraph.
    Returns:
        A NetworkX DiGraph representing the web, with Knots as nodes.
    """

    # lazy import
    import networkx as nx

    if not isinstance(knots, (tuple, list)):
        knots = [knots]

    graph = nx.DiGraph()

    def draw_fn(knot_, _, __):
        for i in knot_.input_knots_list:
            graph.add_edge(i, knot_)
        graph.add_node(knot_, label=knot_.layer_name)

    for knot in knots:
        knot.map_web(draw_fn)

    return graph


def draw_web(knots_or_graph, mode='pydot-300', returnval=False):
    """Displays a visual of the web, including the specified :knots: and all 'earlier' knots, in the sense of
    Knot.map_web.__doc__.

    A more comprehensive visual, for example featuring scopes, can of course be made by calling model_web and then
    viewing the result in TensorBoard. (For example via ktools.tb_view.) This function is primarily here as a debugging
    aid for when model_web fails.

    Arguments:
        knots_or_graph: A Knot or tuple or list of Knots, specifying the output(s) of the web to display. May also be a
            NetworkX graph, as produced by Knot.to_networkx or networkx_web.
        mode: The format to display the web in. Due to system differences, some formats may not function on some systems
            Valid options are 'pydot', 'pydot-<num>', where <num> is a natural number, 'svg' or 'basic'.
            Using 'pydot' will create a graph laid out in a hierarchical manner using pydot; this hierarchical layout is
            usually the best. Using 'pydot-<num>', e.g. 'pydot-300', will do the same as 'pydot', except the image will
            have a resolution determined by the given number. Note that both versions of 'pydot' may occasionally
            result in segfaults, due to instability in dot. Using 'svg' will create an SVG image, in the same manner as
            'pydot', and attempt to display it in ImageMagick, by calling the 'display' command. Using 'basic' will
            resort to using the default 'draw' function of NetworkX; this is most likely to work on all platforms, but
            will not have the nice hierarchical layout; the picture tends to be cleaner though.
        returnval: Whether or not to return something. Defaults to False (to avoid cluttering up the REPL when used
            there.)

    Returns:
        If :returnval: is True, and any of the formats other than 'basic' are used, then it will return a bytes string
        representing the image that is displayed.
    """

    # lazy import
    import io
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import tempfile

    if isinstance(knots_or_graph, nx.Graph):
        graph = knots_or_graph
    else:
        graph = to_networkx_web(knots_or_graph)

    if mode == 'svg':
        dot = nx.drawing.nx_pydot.to_pydot(graph)
        img_str = dot.create_svg()
        with tempfile.NamedTemporaryFile(suffix='.svg') as f:
            f.write(img_str)
            tools.shell(f'display {f.name}')
    else:
        fig = plt.figure()
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        if 'pydot' in mode:
            dot = nx.drawing.nx_pydot.to_pydot(graph)
            try:
                _, res = mode.split('-')
            except ValueError:  # too many values to unpack
                img_str = dot.create_png()
            else:
                img_str = dot.create_png(prog=[dot.prog, f'-Gdpi={res}'])
            sio = io.BytesIO()
            sio.write(img_str)
            sio.seek(0)
            img = mpimg.imread(sio)

            ax.imshow(img)
        elif mode == 'basic':
            img_str = b''
            nx.draw(graph, ax=ax, with_labels=True, labels={n: n.layer_name for n in graph.nodes})
        else:
            raise ValueError(f"Specified format '{mode}' not understood.")
        fig.show()

    if returnval:
        return img_str


def example_web():
    """Constructs an example web of Knots.

    Try calling the draw, make, or model methods of either of the return Knots, or draw_web, make_web or model_web on
    both of them at the same time.

    Returns:
        Two Knots, which form outputs to the web.
    """

    a = Knot(keras.layers.Input(shape=(2,), name='input'))
    b = Knot(keras.layers.Dense(3, name='dense1'))(a)
    layer = keras.layers.Dense(3, name='dense2')
    c = Knot(layer)(a)
    d = Knot(keras.layers.Concatenate(name='concat'))([b, c])
    e = Knot(layer)(d)  # reusing layers is fine
    f = Knot(keras.layers.Dense(3, name='dense3'))(d)
    return e, f
