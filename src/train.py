import tensorflow as tf
import incptae as inc



origin = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
lr = tf.placeholder(tf.float32, shape=[])

inc.build_model(origin, True, lr)