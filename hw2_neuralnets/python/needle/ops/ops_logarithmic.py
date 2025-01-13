from typing import Optional

from numpy.core.multiarray import dtype
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp): # COME BACK!
    def __init__(self) -> None:
        self.axes = (1,)

    def compute(self, Z):

        ### BEGIN YOUR SOLUTION
        max_val = (logsumexp(Tensor(Z), axes=self.axes)).realize_cached_data().reshape((Z.shape[0],1))
        return Z - max_val
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        pass
        # Z = node.inputs[0].realize_cached_data()
        # reshape_max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # expanded_input_shape = reshape_max_z.shape
        # diff = array_api.exp(Z -reshape_max_z) # numpy does automatic broadcast
        # total_sum = array_api.broadcast_to(array_api.sum(diff, axis=self.axes, keepdims=True), Z.shape)
        # one = 1 /total_sum  # do we need to watch numerical stability? (shape of Z; numpy supports element wise division)
        # three = array_api.where(array_api.broadcast_to(reshape_max_z, Z.shape) == Z, 1, 0) # should auto broadcast in numpy
        # two = diff + three*(-total_sum)

        # logsumgrad = (one * two + three)#.reshape(expanded_input_shape).broadcast_to(Z.shape)
        # ones = array_api.ones_like(Z)

        # return Tensor(ones - logsumgrad) * out_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = array_api.max(Z, axis=self.axes, keep_dims=False)
        reshape_max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_diff = array_api.sum(array_api.exp(Z -reshape_max_z), axis=self.axes)
        return array_api.log(sum_diff) + max_z

    def gradient(self, out_grad, node):
        z = node.inputs[0]

        # need to identify the dims that were summed out in the forward pass
        incoming_shape = list(z.shape)
        axes = range(len(incoming_shape)) if self.axes is None else self.axes
        for axis in axes:
            incoming_shape[axis] = 1

        # sum and reshape to fit the input dims
        exp_z = summation(exp(z - z.realize_cached_data().max(self.axes, keepdims=True)),self.axes)
        grad = out_grad / exp_z
        grad_exp_z = grad.reshape(incoming_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)