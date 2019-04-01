# ktools (Keras Tools)
Tools for working with Keras. For example:

* `chain_layers`, for chaining layers together without using a `Model`. (Useful to avoid cluttering up TensorBoard!)
* `replace_layer`, for replacing certain layers and rebuilding a `Model`. Should work even on non-`Sequential` models!
* `PeriodicConv1D` and friends, for creating convolutions with periodic padding.
* `residual_layers`, for quickly creating a ResNet (or ResNet-in-ResNet (etc.) architectures.
* `tb_view`, for quickly opening a model in TensorBoard.
