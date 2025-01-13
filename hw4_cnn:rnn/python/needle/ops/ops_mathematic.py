
"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


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

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # return array_api.power(a,b)
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        dout_da = b * (a ** (b-1))
        dout_db = log(a) * ((a ** b))
        # dout_db = array_api.log(a) * ((a ** b))
        return out_grad*dout_da, out_grad*dout_db
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # return array_api.power(a, self.scalar)
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        dout_da = self.scalar * (a ** (self.scalar-1))
        return out_grad*dout_da
        # dout_da = multiply(self.scalar, power_scalar(a, self.scalar-1))
        # return multiply(out_grad, dout_da)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        print(f"What type is array_api? {array_api}")
        print(f"inside compute ewise div: {type(a)}")
        print(f"inside compute ewise div: {type(b)}")
        return_val =  a/b
        print(f"inside compute ewise div: {type(return_val)}")
        return return_val
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        return out_grad / b, - a * out_grad / b ** 2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.divide(a,self.scalar, dtype=a.dtype)
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
  # this assumes input is 2d; in general case we need to flip the last dim
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if axes else None

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        n = len(a.shape)
        idxs = list(range(n))
        if not self.axes: 
          self.axes = tuple([n-1, n-2])
        assert len(self.axes) == 2, "incorrect assumption"
        idxs[self.axes[0]], idxs[self.axes[1]] = idxs[self.axes[1]], idxs[self.axes[0]]
        return a.permute(tuple(idxs))
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
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.compact().reshape(node.inputs[0].shape)
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
        start_dims = node.inputs[0].shape
        end_dims = self.shape
        candidates = [i for i in range(len(end_dims))]
        for i, (start, end) in enumerate(zip(reversed(start_dims), reversed(end_dims))):
          if start == end:
            candidates[len(end_dims) -1 - i] = -1
        new_dims = tuple(filter(lambda x: x != -1, candidates)) 
        return out_grad.sum(new_dims).compact().reshape(start_dims)
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
        ### BEGIN YOUR SOLUTION --> None != 0 in python!
        new_shape = list(node.inputs[0].shape)
        if self.axes == None:
            summation_dims = list(range(len(new_shape)))
        elif isinstance(self.axes, tuple) or isinstance(self.axes, list):
            summation_dims = self.axes
        else:
            summation_dims = [self.axes]
        for axis in summation_dims: # identify axes that we summed out in the forward computation
            new_shape[axis] = 1        
        return out_grad.reshape(tuple(new_shape)).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.matmul(a,b)
        return a @ b
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
        return -a # overloaded with __neg__
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
        out = node.realize_cached_data().copy()
        out[out > 0] = 1 ### dont think ndarray supports this conditional slice
        return out_grad * Tensor(out, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = -(tanh(a)**2) 
        grad = grad + 1
        return grad * out_grad 
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        # compute the shape of the final array
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))

        # make an empty array with this shape
        new_array = array_api.empty(shape=tuple(new_shape),device=args[0].device)
        slices = [slice(0, shape, 1) for shape in new_shape]

        # loop over the arrays in the tuple, copy them into each splice
        for num, tensor in enumerate(args):
            # construct splice that is all dims except for num in the axis location
            slices[self.axis] = slice(num,num+1,1) # want to extract a single entry at once
            new_array[tuple(slices)] = tensor

        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return_val = split(out_grad, self.axis)
        return return_val
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        
        # get the shape of the tensor
        new_shape = list(A.shape) # (5,5,3)
        new_shape.pop(self.axis)
        old_shape = list(A.shape)

        slices = [slice(0, shape, 1) for shape in old_shape]
        sub_arrays = [] # init empty list that will contain splices along particular dim

        for mode in range(old_shape[self.axis]):
            # get splice for mode
            slices[self.axis] = slice(mode, mode+1, 1)
            sub_arrays.append(A[tuple(slices)].compact().reshape(tuple(new_shape)))
        return tuple(sub_arrays)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print(f"inside flip forward: {type(a)}")
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(f"inside flip backward: {type(out_grad)}")
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        # need to make a new array which has a mutuated dim
        new_shape = list(a.shape) # a is a NDarray
        for axis in self.axes: # expanding out dimentions that we are going to dialate along
            new_shape[axis] *= (self.dilation + 1) 
        
        # make shell of larger array
        dialted_array = array_api.full(shape=tuple(new_shape),fill_value=0,device=a.device)
        
        # make slices into the larger array
        slices = []
        for i in range(len(new_shape)):
            if i in self.axes:
                slices.append(slice(0,new_shape[i], self.dilation+1))  
            else:
                slices.append(slice(0,new_shape[i], 1))  
        
        # slice in copy
        dialted_array[tuple(slices)] = a
        return dialted_array
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        # need to make a new array which has reduced dims along shape
        new_shape = list(a.shape) # a is a NDarray
        for axis in self.axes: # expanding out dimentions that we are going to dialate along
            new_shape[axis] //= (self.dilation + 1) 
        
        # make shell of larger array
        undialted_array = array_api.full(shape=tuple(new_shape), fill_value=0, device=a.device)
        
        # make slices into the larger array
        slices = []
        for i in range(len(new_shape)):
            if i in self.axes:
                slices.append(slice(0,a.shape[i], self.dilation+1))  
            else:
                slices.append(slice(0,a.shape[i], 1))  
        
        # slice in copy
        undialted_array = a[tuple(slices)]
        return undialted_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A_padded = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        
        # extract strides from input
        N_s,H_s,W_s,C_s = A_padded.strides

        # assuming that A is the input B x H x W x C and B is weighs K x K x C x num_filters
        N,H,W,C_in = A_padded.shape
        K,_,_,num_filters = B.shape
        
        # reshape A so that each elementwise product is a row
        H_out = (H - K + 1)//self.stride
        W_out = (W - K + 1)//self.stride
        
        # allocate new memory for conv computation (see notebook for stride formula !)
        big_A = A_padded.as_strided(shape=(N,H_out,W_out,K,K,C_in),
                              strides = (N_s,H_s*self.stride,W_s*self.stride,H_s,W_s,C_s)) \
                              .compact().reshape((N*H_out*W_out, K*K*C_in))
        big_B = B.compact().reshape((K*K*C_in, num_filters))
        big_out = big_A @ big_B
        return big_out.compact().reshape((N,H_out,W_out,num_filters))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)