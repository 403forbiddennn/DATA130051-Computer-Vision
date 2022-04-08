from builtins import range
import numpy as np


def affine_forward(x, w, b):
    flatten_x = x.reshape((x.shape[0], -1))
    out = np.dot(flatten_x, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    flatten_x = x.reshape((x.shape[0], -1))
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(flatten_x.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def softmax_loss(x, y):
    num_train = x.shape[0]

    f = x - np.max(x, axis=1).reshape(num_train, 1)
    softmax = np.exp(f) / np.sum(np.exp(f), axis=1).reshape(num_train, 1)
    loss = np.sum(-np.log(softmax[np.arange(num_train), y]))

    softmax[np.arange(num_train), y] -= 1
    dx = softmax

    loss /= num_train
    dx /= num_train

    return loss, dx

