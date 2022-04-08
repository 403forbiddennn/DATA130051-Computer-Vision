import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    m, v, t, beta1, beta2, eps, learning_rate = config['m'], config['v'], config['t'], config['beta1'], \
                                                config['beta2'], config['epsilon'], config['learning_rate']
    t += 1
    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    vt = v / (1 - beta2 ** t)
    w -= learning_rate * mt / (np.sqrt(vt) + eps)
    config['m'], config['v'], config['t'] = m, v, t
    next_w = w

    return next_w, config