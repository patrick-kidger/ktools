from .src.activation import (concat_multiple_activations,
                             minus_activation,
                             concat_activation,
                             cleaky_relu,
                             celu,
                             cselu,
                             softthresh,
                             QuasiIdentity)

from .src.datagen import (TransformedSequence,
                          batch_generator,
                          MultiprocessGenerator)

from .src.initializers import NearIdentity

from .src.layers import (ChainLayers,
                         chain_layers,
                         Periodize,
                         PeriodicConv1D,
                         PeriodicConv2D,
                         PeriodicSeparableConv1D,
                         PeriodicSeparableConv2D,
                         PeriodicDepthwiseConv2D,
                         PeriodicConv2DTranspose,
                         PeriodicConv3D,
                         PeriodicConv3DTranspose,
                         replace_layers,
                         dense_block,
                         dense_change_size,
                         residual_layers)

from .src.misc import (WithTrainable,
                       uniq_name,
                       periodize,
                       periodic_convolve)

from .src.scopes import (get_name_scope,
                         get_current_scopes)

from .src.visualise import (tb_view,
                            plot_fn,
                            plot_model_history)

from .src.web import (Knot,
                      CycleException,
                      model_web,
                      deepcopy_web,
                      draw_web,
                      example_web)
