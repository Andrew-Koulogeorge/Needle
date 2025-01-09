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
        # Z = node.inputs[0].realize_cached_data()
        pass
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
        ### BEGIN YOUR SOLUTION

        max_z = array_api.max(Z, axis=self.axes)
        reshape_max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_diff = array_api.sum(array_api.exp(Z -reshape_max_z), axis=self.axes)
        return array_api.log(sum_diff) + max_z

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION
        # ### BEGIN YOUR SOLUTION

        # # print("Inside the backward function for logsumexp")
        # # print(f"Outgrad value for the backward function for logsumexp: {out_grad.dtype}")

        # Z = node.inputs[0].realize_cached_data()

        # # print(f"Inputs Cache value for the backward function for logsumexp: {Z.dtype}")

        # reshape_max_z = array_api.max(Z, axis=self.axes, keepdims=True)

        # # print(f"maxed value for the backward function for logsumexp: {reshape_max_z.dtype}")

        # expanded_input_shape = reshape_max_z.shape
        # diff = array_api.exp(Z -reshape_max_z, dtype=Z.dtype) # numpy does automatic broadcast

        # # print(f"exp value for the backward function for logsumexp: {diff.dtype}")

        # total_sum = array_api.broadcast_to(array_api.sum(diff, axis=self.axes, keepdims=True), Z.shape)
        # one = array_api.reciprocal(total_sum, dtype=Z.dtype)  # do we need to watch numerical stability? (shape of Z; numpy supports element wise division)
        # three = array_api.where(array_api.broadcast_to(reshape_max_z, Z.shape) == Z, 1, 0).astype(Z.dtype) # should auto broadcast in numpy
        
        # # print(f"three value for the backward function for logsumexp: {three.dtype}")
        # # print(f"diff value for the backward function for logsumexp: {diff.dtype}")
        # # print(f"total_sum value for the backward function for logsumexp: {total_sum.dtype}")
        # two = diff + three*(-total_sum)

        # # print(f"two value for the backward function for logsumexp: {two.dtype}")
        
        # return Tensor((one * two + three)) * out_grad.reshape(expanded_input_shape).broadcast_to(Z.shape)
        # ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

