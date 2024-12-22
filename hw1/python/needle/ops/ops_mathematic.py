"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from numpy.core.multiarray import array

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad # adjoint_a, adjoint_b


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    # assuming we want to raise a ** b element wise
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        
        a,b = node.inputs
        dout_da = multiply(b, power(a, b-1))
        dout_db = multiply(log(a), power(a, b))

        return multiply(out_grad, dout_da), multiply(out_grad, dout_db)

        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad:Tensor, node:Tensor):
        ### BEGIN YOUR SOLUTION

        # a = node.inputs[0]
        # dout_da = multiply(self.scalar, power_scalar(a, self.scalar-1))
        # return multiply(out_grad, dout_da)

        return out_grad * self.scalar * (node.inputs[0] ** self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad:Tensor, node:Tensor):
        ### BEGIN YOUR SOLUTION
        # why is this not working for the first but working for the second?

        a,b = node.inputs
        # dout_da = divide(1,b)
        # dout_db = negate(divide(a,power_scalar(b,2)))      
        # return multiply(out_grad, dout_da), multiply(out_grad, dout_db)

        
        return out_grad / b, - a * out_grad / b ** 2
        
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if axes != None else (-2,-1)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # identify the dimentions that dont match
        start_dims = node.inputs[0].shape
        end_dims = self.shape
        candidates = [i for i in range(len(end_dims))]

        for i, (start, end) in enumerate(zip(reversed(start_dims), reversed(end_dims))):
          if start == end:
            candidates[len(end_dims) -1 - i] = -1
        new_dims = tuple(filter(lambda x: x != -1, candidates)) # remove the matching dims
        
        # sum them out, reshape and return
        return out_grad.sum(new_dims).reshape(start_dims)


        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # compare original dims with the sum dims and add 1 into shapes
        # then broadcast the out_grad to match the same shape as the ingrad?
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        left = right = 0
        
        input_len = len(input_shape)
        output_len = len(output_shape)
        new_dim = []
        
        while left < input_len and right < output_len:
          if input_shape[left] == output_shape[right]:
            new_dim.append(input_shape[left])
            left += 1
            right += 1
          else:
            new_dim.append(1)
            left += 1
        while left < input_len:
            new_dim.append(1)
            left += 1
        
        return out_grad.reshape(tuple(new_dim)).broadcast_to(input_shape)

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        dout_da, dout_db = out_grad @ transpose(b), transpose(a) @ out_grad

        # if we have batched solutions, need to enure that gradient shape matches that of the input arguments
        if(len(a.shape) < len(b.shape)): # broadcasting a over b
          dout_da = summation(dout_da, tuple([i for i in range(len(b.shape)-len(a.shape))]))
        elif (len(b.shape) < len(a.shape)): # broadcasting b over a
          dout_db = summation(dout_db, tuple([i for i in range(len(a.shape)-len(b.shape))]))

        return dout_da, dout_db
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.multiply(a, -1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raw_incoming_data = node.inputs[0].realize_cached_data()
        mask = array_api.where(raw_incoming_data >= 0, 1, 0)
        return Tensor(mask) * out_grad

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

