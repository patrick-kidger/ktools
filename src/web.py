import io
import ktools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import tensorflow as tf
import tensorflow.keras as keras
import tools


_sentinel = object()  # Used to detect cycles in webs


class Knot:
    """Represents the vertex of a graph. Here we instead call them knots in webs, however, to avoid confusion with the
    usual TensorFlow meaning of 'Graph'.

    Must have a layer attached to it, and it may be connected to other knots, which will be treated as its inputs.
    Calling make() on a Knot will then call its layer on the layers of inputs Knots (recursively).

    Whilst Keras does have Nodes which  keeps track of something similar, they don't have quite the flexibility that
    we're after to make this work.
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

    def clone(self, include_inputs=False):
        cloned_self = self.__class__(self.layer, data=self.data)
        if include_inputs:
            cloned_self(self.input_knots)
        cloned_self.set_scope_names(self.scope_names)
        return cloned_self

    @property
    def input_knots_list(self):
        if self._input_knots is None:
            return []
        elif isinstance(self._input_knots, (tuple, list)):
            return self._input_knots
        else:
            return [self._input_knots]

    @property
    def input_knots(self):
        return self._input_knots

    @property
    def layer(self):
        return self._layer

    def set_layer(self, layer):  # somehow this seems nicer than using property.setter.
        self._layer = layer

    def set_scope_names(self, scope_names):
        if scope_names and scope_names[-1] != '/':
            scope_names += '/'
        self.scope_names = scope_names

    def register_current_scopes(self):
        self.set_scope_names(ktools.get_current_scopes())

    def prepend_current_scopes(self):
        pieces = [ktools.get_current_scopes()]
        if self.scope_names:  # in particular not '' or None
            pieces.append(self.scope_names)
        self.set_scope_names('/'.join(pieces))

    def _add_output_knot(self, knot):
        self._output_knots.add(knot)

    def _remove_output_knot(self, knot):
        self._output_knots.remove(knot)

    def __call__(self, input_):
        if isinstance(input_, (tuple, list)):
            assert len(input_) > 0
            for inp in input_:
                assert isinstance(inp, Knot)
        else:
            assert isinstance(input_, Knot)

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
        self.replace_multi([knot], knot)

    def replace_multi(self, input_knots, output_knot):
        if not isinstance(input_knots, (tuple, list)):
            input_knots = [input_knots]
        if self.input_knots:
            # have the input knots inherit our input, and have the input depend on it.
            for knot in input_knots:
                knot(self.input_knots)
        for inp in self.input_knots_list:
            # noinspection PyProtectedMember
            inp._remove_output_knot(self)
        for n in self._output_knots:  # update our outputs to depend on the new output knot instead
            if isinstance(n._input_knots, (tuple, list)):
                n._input_knots = [output_knot if x is self else x for x in n._input_knots]
            else:
                n._input_knots = output_knot

    def map_web(self, fn, memodict=None):
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
        return model_web(self, input_knots)

    def __deepcopy__(self, memodict=None):
        return deepcopy_web(self, memodict)

    def deepcopy(self, memodict=None):
        return self.__deepcopy__(memodict)

    def draw(self, pydot=True, res=300):
        # noinspection PyTypeChecker
        return draw_web(self, pydot=pydot, res=res)


class CycleException(tools.DefaultException):
    default_msg = 'Cycle detected in web.'


def model_web(output_knots, input_knots=None):
    if input_knots is not None:
        if not isinstance(input_knots, (tuple, list)):
            input_knots = [input_knots]
        for input_knot in input_knots:
            assert not input_knot.input_knots_list  # Inputs can't have inputs!
    if not isinstance(output_knots, (tuple, list)):
        output_knots = [output_knots]

    # Use the same memodict for all of them, as else each output tensor will exist in a disjoint subgraph to the others
    memodict = {}
    outputs = [output_knot.make(memodict=memodict) for output_knot in output_knots]
    inputs = memodict['__inputs']
    if input_knots is not None:
        assert inputs == {input_knot.layer for input_knot in input_knots}
    inputs = list(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def deepcopy_web(knots, memodict=None):
    def deepcopy_fn(knot, copied_inputs, _):
        # don't copy the layers and such
        new_knot = knot.clone()
        # do deepcopy our input knots
        if copied_inputs is not None:
            new_knot(copied_inputs)
        return new_knot

    if isinstance(knots, (tuple, list)):
        return [knot.map_web(deepcopy_fn, memodict) for knot in knots]
    else:
        return knots.map_web(deepcopy_fn, memodict)


def draw_web(knots, pydot=True, res=300):
    if not isinstance(knots, (tuple, list)):
        knots = [knots]

    graph = nx.DiGraph()

    def draw_fn(knot_, _, __):
        for i in knot_.input_knots_list:
            graph.add_edge(i, knot_)
        graph.add_node(knot_, label=knot_.layer.name)

    for knot in knots:
        knot.map_web(draw_fn)

    if pydot:
        dot = nx.drawing.nx_pydot.to_pydot(graph)
        png_str = dot.create_png(prog=[dot.prog, f'-Gdpi={res}'])
        sio = io.BytesIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)

        fig = plt.figure()
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.show()
    else:
        nx.draw(graph, with_labels=True, labels={n: n.layer.name for n in graph.nodes})
        plt.show(block=False)
    return graph


def example_web():
    a = Knot(tools.Object(name='one'))
    b = Knot(tools.Object(name='two'))(a)
    c = Knot(tools.Object(name='three'))(a)
    d = Knot(tools.Object(name='four'))([b, c])
    e = Knot(tools.Object(name='three'))(d)  # deliberate duplicate name to demonstrate that we can have them
    f = Knot(tools.Object(name='five'))(d)
    return e, f
