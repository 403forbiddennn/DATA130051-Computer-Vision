import numpy as np
import os
import gzip


def load_data(root_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []

    for file_name in files:
        paths.append(os.path.join(root_folder, file_name))

    with gzip.open(paths[0], 'rb') as label_path:
        y_train = np.frombuffer(label_path.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as image_path:
        X_train = np.frombuffer(image_path.read(), np.uint8, offset=16).reshape((len(y_train), 28, 28))

    with gzip.open(paths[2], 'rb') as label_path:
        y_test = np.frombuffer(label_path.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as image_path:
        X_test = np.frombuffer(image_path.read(), np.uint8, offset=16).reshape((len(y_test), 28, 28))

    return X_train, y_train, X_test, y_test


def split_train_val(X_data, y_data, val_ratio):
    np.random.seed(89)
    shuffled_indices = np.random.permutation(X_data.shape[0])
    val_set_size = int(X_data.shape[0] * val_ratio)
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    X_train, y_train = X_data[train_indices], y_data[train_indices]
    X_val, y_val = X_data[val_indices], y_data[val_indices]
    return X_train, y_train, X_val, y_val

