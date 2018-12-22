import os

import numpy as np
from scipy.misc import imsave
from conf import Config as conf


def load_from_file(path):
    fd = open(path)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    return loaded

def load_mnist(is_training=True):
    loaded = load_from_file(os.path.join(conf.dataset, 'train-images-idx3-ubyte'))
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 'train-labels-idx1-ubyte'))
    trY = loaded[8:].reshape((60000)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 't10k-images-idx3-ubyte'))
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 't10k-labels-idx1-ubyte'))
    teY = loaded[8:].reshape((10000)).astype(np.float)

    if is_training:
        return trX / 255, trY
    else:
        return teX / 255., teY
        
def shuffle(data, label):
    p = np.random.permutation(data.shape[0])
    return data[p], label[p]
    
def merge_img(data, size):
    if (len(data.shape) == 4):
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
        
    res = np.zeros((data.shape[1]*size[0], data.shape[2]*size[1]), dtype=np.float32)
    count = 0
    for i in range(size[0]):
        for j in range(size[1]):
            res[i*data.shape[1]:(i+1)*data.shape[1], j*data.shape[2]:(j+1)*data.shape[2]] = data[count]
            count += 1
    return res

def save_img(data, filepath):
    imsave(filepath, data*255)
    

if __name__ == '__main__':
    data, label = load_mnist()
    print(data.shape)
    print(label.shape)
    
    res = merge_img(data, [10, 10])
    save_img(res, 'res.png')