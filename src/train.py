import numpy as np
import tensorflow as tf
import incptae as inc

from utils import *
from conf import Config as conf

def train():
    # build incptae
    X = tf.placeholder(tf.float32, shape=(conf.batch_size, conf.data_size, conf.data_size, 1))
    loss, opt, reconst = inc.build_model(X, True, conf.learning_rate)
    
    init = tf.global_variables_initializer()
    
    data, label = load_mnist()
    
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(init)
        for i in range(conf.training):
            this_data, this_label = shuffle(data, label)
            for j in range(this_data.shape[0] // conf.batch_size):
                train_X = this_data[conf.batch_size*j:conf.batch_size*(j+1)]
                l, _, res = sess.run([loss, opt, reconst], feed_dict={X:train_X})
                if (j % 20 == 0):
                    print('[%d]loss: %.4f'%(i, l))
                """
                if (j % 100 == 0):
                    res = merge_img(res, [4, 4])
                    save_img(res, os.path.join(conf.img_dir, 'reconst_'+str(j)+'.png'))
                    res = merge_img(train_X, [4, 4])
                    save_img(res, os.path.join(conf.img_dir, 'input_'+str(j)+'.png'))
                """

if __name__ == '__main__':
    train()