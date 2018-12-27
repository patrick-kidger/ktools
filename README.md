# ktools (Keras Tools)
Tools for working with Keras. For example:

* `chain_layers`, for chaining layers together without using a `Model`. (Useful to avoid cluttering up TensorBoard!)
* `replace_layer`, for replacing certain layers and rebuilding a `Model`. Should work even on non-`Sequential` models!
* `get_variable_scopes`, for getting all of the scopes that a layer was called in.
* `tb_view`, for quickly opening a model in TensorBoard.
