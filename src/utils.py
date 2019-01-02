import os

import numpy as np
from scipy.misc import imsave
from conf import Config as conf
import pickle

import tensorflow as tf

# load cifar file
def read_cifar(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# convert cifar binary into image
def form_img(batch_data):
    """
    batch_data: [batch_size, 1024 + 1024 + 1024 = 3072(rgb)]
    return: [batch_size, 1024, 3]
    """
    res = batch_data.reshape((batch_data.shape[0], 3, 32, 32))
    res = res.transpose([0, 2, 3, 1])
    return res

# api for cifar loading
def load_cifar(is_training=True):
    # path = os.path.join('data', 'mnist')
    path = conf.cifar_dir
    if is_training:
        
        data1 = read_cifar(os.path.join(path, 'data_batch_1'))
        data2 = read_cifar(os.path.join(path, 'data_batch_2'))
        data3 = read_cifar(os.path.join(path, 'data_batch_3'))
        data4 = read_cifar(os.path.join(path, 'data_batch_4'))
        
        cifar = np.concatenate((form_img(data1[b'data']), form_img(data2[b'data']), form_img(data3[b'data']), form_img(data4[b'data'])), axis=0)
        print(cifar.shape)

        label = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels']), axis=0) 

        return cifar / 255., label

    else:
        data5 = read_cifar(os.path.join(path, 'data_batch_5'))
        teX = form_img(data5[b'data'])
        teY = np.array(data5[b'labels'])
        return teX / 255.0, teY

"""
# function for tfrecord
def _generate_sharded_filenames(data_dir):
  base_name, num_shards = data_dir.split("@")
  num_shards = int(num_shards)
  file_names = []
  for i in range(num_shards):
    file_names.append('{}-{:0>5d}-of-{:0>5d}'.format(base_name, i, num_shards))
  return file_names

def load_tfrecord(filename_queue, image_dim=28, split='train'):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_dim, image_dim, 1])
    image.set_shape([image_dim, image_dim, 1])

    image = tf.cast(image, tf.float32) * (1. / 255)
    
    label = tf.cast(features['label'], tf.int32)
    features = {
        'images': image,
        'labels': label,
    }

    return features

def load_mnist_tfrecord(is_training=True):
    file_format = '{}_{}shifted_mnist.tfrecords'
    if is_training:
        shift = 2
        split = 'train'
    else:
        shift = 0
        split = 'test'

    filenames = [os.path.join(conf.data_dir, file_format.format(split, shift))]

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, shuffle=(split == 'train'))

        features = load_tfrecord(filename_queue, split=split)
        if is_training:
            batched_features = tf.train.shuffle_batch(
                features,
                batch_size=conf.batch_size,
                num_threads=2,
                capacity=conf.batch_capacity+3*conf.batch_size,
                min_after_dequeue=conf.batch_capacity) 
        else:
            batched_features = tf.train.batch(
                features,
                batch_size=conf.batch_size,
                num_threads=1,
                capacity=conf.batch_capacity+3*conf.batch_size)

    return batched_features['images'], batched_features['labels']
"""

def load_from_file(path):
    fd = open(path)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    return loaded

def load_mnist(is_training=True):
    path = conf.mnist_dir
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

def load_fashion(is_training=True):
    path = conf.fashion_dir
    loaded = load_from_file(os.path.join(dataset, 'train-images-idx3-ubyte'))
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(dataset, 'train-labels-idx1-ubyte'))
    trY = loaded[8:].reshape((60000)).astype(np.float)

    loaded = load_from_file(os.path.join(dataset, 't10k-images-idx3-ubyte'))
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(dataset, 't10k-labels-idx1-ubyte'))
    teY = loaded[8:].reshape((10000)).astype(np.float)

    if is_training:
        return trX / 255, trY
    else:
        return teX / 255., teY
        
def shuffle(data, label):
    p = np.random.permutation(data.shape[0])
    return data[p], label[p]
    
def merge_img(data, size):
    if (len(data.shape) == 4 and data.shape[3] == 1):
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    
        res = np.zeros((data.shape[1]*size[0], data.shape[2]*size[1]), dtype=np.float32)
        count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                res[i*data.shape[1]:(i+1)*data.shape[1], j*data.shape[2]:(j+1)*data.shape[2]] = data[count]
                count += 1
    else:
        res = np.zeros((data.shape[1]*size[0], data.shape[2]*size[1], 3), dtype=np.float32)
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
