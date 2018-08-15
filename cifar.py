import pickle
import sys
import tensorflow as tf

import numpy as np

import os

from scipy.misc import imsave

def read_cifar(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def form_img(batch_data):
    """
    batch_data: [batch_size, 1024 + 1024 + 1024 = 3072(rgb)]
    return: [batch_size, 1024, 3]
    """
    res = batch_data.reshape((batch_data.shape[0], 3, 32, 32))
    res = res.transpose([0, 2, 3, 1])
    return res

def store_img(batch_data, label):
    for i in range(1, 100):
        img = batch_data[i]
        print(img.shape)
        imsave(os.path.join('./img/'+str(label[i]), str(i) + '.png'), img)

def load_cifar(batch_size, is_training=True):
    # path = os.path.join('data', 'mnist')
    path = './data/cifar-10-python/cifar-10-batches-py'
    if is_training:
        
        data1 = read_cifar(os.path.join(path, 'data_batch_1'))
        data2 = read_cifar(os.path.join(path, 'data_batch_2'))
        data3 = read_cifar(os.path.join(path, 'data_batch_3'))
        data4 = read_cifar(os.path.join(path, 'data_batch_4'))
        
        cifar = np.concatenate((form_img(data1[b'data']), form_img(data2[b'data']), form_img(data3[b'data']), form_img(data4[b'data'])), axis=0)
        print(cifar.shape)

        label = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels']), axis=0) 
        print(label.shape)

        trX = cifar
        trY = label

        trX = tf.convert_to_tensor(trX/255., tf.float32)
        trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)

        return trX, trY
    else:
        data5 = read_cifar(os.path.join(path, 'data_batch_5'))
        teX = form_img(data5[b'data'])
        teY = np.array(data5[b'labels'])

        teX = tf.convert_to_tensor(teX/255., tf.float32)
        teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

        return teX, teY

if __name__ == '__main__':

    """
    if not os.path.exists('./img'):
        os.makedirs('./img')
    for i in range(10):
        if not os.path.exists(os.path.join('./img', str(i))):
            os.makedirs(os.path.join('./img', str(i)))
    res = read_cifar(sys.argv[1])
    print(res.keys())
    print(res[b'data'].shape)

    imgset = form_img(res[b'data'])
    store_img(imgset, res[b'labels'])
    """
    trX, trY, valX, valY = load_cifar(128)
    tstX, tstY = load_cifar(128, False)

    print(trX.shape, trY.shape, valX.shape, valY.shape)
    print(tstX.shape, tstY.shape)

