# Code for "Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking"

Paper Link: TBA

Code:
* `train_mod_add.py`: Train a two-layer ReLU net on modular addition
* `train_mod_add_nowd.py`: Train a two-layer ReLU net on modular addition without weight decay. A special learning rate schedule is applied to speed up the training in the late phase.
* `train_diag_cls.py`: Train a diagonal linear net on sparse linear classification.
* `train_diag_cls2.py`: Train a diagonal linear net on linear classification, where the data has a very large L2 margin.
* `train_mc.py`: Optimize for an overparameterized matrix completion problem.