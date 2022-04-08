from builtins import range
from builtins import object
from project_tools import layers
import numpy as np


class TwoLayerNet(object):
    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.params = {
            'W1': weight_scale * np.random.randn(input_dim, hidden_dim),
            'b1': np.zeros(hidden_dim),
            'W2': weight_scale * np.random.randn(hidden_dim, num_classes),
            'b2': np.zeros(num_classes)
        }

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        out1, cache1 = layers.affine_relu_forward(X, W1, b1)
        scores, cache2 = layers.affine_forward(out1, W2, b2)

        if y is None:
            return scores

        loss, dout = layers.softmax_loss(scores, y)
        loss += (0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2))
        dout1, dW2, db2 = layers.affine_backward(dout, cache2)
        _, dW1, db1 = layers.affine_relu_backward(dout1, cache1)
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        return loss, grads

