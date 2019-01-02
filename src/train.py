import numpy as np
import tensorflow as tf
import incptae as inc

from utils import *
from conf import Config as conf

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

def train():
    # build incptae
    X = tf.placeholder(tf.float32, shape=(conf.batch_size, conf.data_size[0], conf.data_size[1], conf.data_size[2]))
    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(conf.learning_rate, global_steps, 2000, 0.96)
    loss, opt, reconst = inc.build_model(X, True, global_steps, learning_rate)

    saver = tf.train.Saver(max_to_keep=20)
    init = tf.global_variables_initializer()

    if conf.dataset == 'mnist':
        data, label = load_mnist()
        assert conf.data_size == [28, 28, 1]
    elif conf.dataset == 'fashion':
        data, label = load_fashion()
        assert conf.data_size == [28, 28, 1]
    else:
        data, label = load_cifar()
        assert conf.data_size == [32, 32, 3]
    
    with tf.Session(config=tf.ConfigProto()) as sess:
        try:
            checkpoint = tf.train.latest_checkpoint(conf.model_dir)
            saver.restore(sess, checkpoint)
        except:
            sess.run(init)

        for i in range(conf.training):
            this_data, this_label = shuffle(data, label)

            for j in range(this_data.shape[0] // conf.batch_size):
                train_X = this_data[conf.batch_size*j:conf.batch_size*(j+1)]
                l, _, res, gs = sess.run([loss, opt, reconst, global_steps], feed_dict={X:train_X})
                if (j % 100 == 0):
                    print('[%d/%d]loss: %.4f'%(gs, i, l))

            # save reconstruct image and input
            res = merge_img(res, [4, 4])
            save_img(res, os.path.join(conf.img_dir, 'reconst_'+str(gs)+'.png'))
            res = merge_img(train_X, [4, 4])
            save_img(res, os.path.join(conf.img_dir, 'input_'+str(gs)+'.png'))
            # save model
            saver.save(sess, os.path.join(conf.model_dir, 'model-'+str(gs)+'.ckpt'))

if __name__ == '__main__':
    train()
