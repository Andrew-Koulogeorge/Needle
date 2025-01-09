"""Optimization module"""
from typing import DefaultDict
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = DefaultDict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # dont need to track gradients for parameter updates
        # compute new update direction baesd on:
        # (1) current gradients of parameters
        # (2) exponential moving average of previous gradients

        # # loop over all the parameters?

        for param in self.params:
          parameter_id = id(param) # cache id
          
          # compute weight decay
          grad = param.grad.data + self.weight_decay*param.data if self.weight_decay > 0 else param.grad.data
          
          # compute gradient direction
          self.u[parameter_id] = self.momentum*self.u[parameter_id] + (1-self.momentum)*grad
          
          # update parameter
          param.data -= self.lr * self.u[parameter_id]

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = DefaultDict(float)
        self.v = DefaultDict(float)


    def step(self):
        ### BEGIN YOUR SOLUTION
        # update step number after all parameters have been updated
        self.t += 1
        # loop over parameters
        for param in self.params:
          param_id = id(param)
          # apply weight decay if flag set
          if self.weight_decay:
            grad = param.grad.data + self.weight_decay*param.data
          else:
            grad = param.grad.data
        
          # compute gradient direction 
          self.u[param_id] = ((1-self.beta1)*grad + (self.beta1)*self.u[param_id])
          unbiased_grad = self.u[param_id]/(1-(self.beta1**self.t))

          # compute adaptive learning rate
          self.v[param_id] = ((1-self.beta2)*(grad**2) + (self.beta2)*self.v[param_id])
          unbiased_norm = self.v[param_id]/(1-(self.beta2**self.t))

          # take gradient step
          param.data -= self.lr*unbiased_grad/((unbiased_norm ** (1/2)) + self.eps)
