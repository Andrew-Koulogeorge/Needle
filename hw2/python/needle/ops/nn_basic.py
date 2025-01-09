"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        
        # init the weight matrix 
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype, requires_grad=True))
        
        # init the bias term
        self.bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype, requires_grad=True).transpose()) if bias else None

        if self.bias:
            assert self.bias.shape == (1,out_features)

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X @ self.weight
        if self.bias: # some layers may not imlpement bias
            output += self.bias.broadcast_to(output.shape)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules # assuming that this module list is valid

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for mod in self.modules: 
            x = mod(x) # tensor in, tensor out; each module must have a well defined __call__ implemented
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n, k = logits.shape
        val = ops.logsumexp(logits, axes=(1,)).reshape((n,1)).broadcast_to(logits.shape)
        one_hot = init.one_hot(k, y) # n x k
        total = ((val - logits)*one_hot).sum()

        # print(type(n))
        # print(total.dtype)
        return_val = total/n # overloading element wise divide operator 
        # print(return_val.dtype)
        

        return return_val
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION 
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n,d = x.shape

        # need to check if its training or evaluation mode
        if self.training:
          mean = (x.sum(axes=(0,))/n)
          var = (((x - mean.broadcast_to((n,d))) ** 2).sum(axes=(0,))/n)
          
          # update running
          self.running_mean = self.momentum * mean + (1 - self.momentum)*self.running_mean.data
          self.running_var = self.momentum * var + (1 - self.momentum)*self.running_var.data          
          
          normalized = (x - mean.broadcast_to((n,d)))/((var.broadcast_to((n,d)) + self.eps)**(1/2))
          
          return self.weight.broadcast_to((n,d))*normalized + self.bias.broadcast_to((n,d))
        else:
          normalized = (x - self.running_mean.broadcast_to(x.shape))/((self.running_var.broadcast_to(x.shape) + self.eps)**(1/2))
          # return normalized
          return self.weight.broadcast_to((n,d))*normalized + self.bias.broadcast_to((n,d))

        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(dim))
        self.b = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n,d = x.shape
        mean = (x.sum(axes=(1,))/d).reshape((n,1)).broadcast_to((n,d))
        var = (((x - mean) ** 2).sum(axes=(1,))/d).reshape((n,1)).broadcast_to((n,d))
        normalized = (x - mean)/((var + self.eps)**(1/2))
        return self.w.reshape((1,d)).broadcast_to((n,d))*normalized + self.b.reshape((1,d)).broadcast_to((n,d))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n,d = x.shape
        if self.training:
          mask = init.randb(n,d, p=1-self.p)
          return mask * x / (1-self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
