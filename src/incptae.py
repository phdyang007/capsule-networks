import tensorflow as tf
import tensorflow.contrib.slim as slim


def incept_encoder(net, is_training=True, reuse=tf.AUTO_REUSE, scope='ie'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], 
                            activation_fn=tf.nn.relu, 
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(net, 64, [7, 7], stride=2, scope='conv1')
            print(net.get_shape())
            net = slim.conv2d(net, 128, [7, 7], stride=2, scope='conv2')
            print(net.get_shape())
            incpt1 = slim.conv2d(net, 32, [1, 1], stride=1, scope='incpt1')
            print(incpt1.get_shape())
            incpt3_1 = slim.conv2d(net, 64, [3, 3], stride=1, scope='incpt3_1')
            print(incpt3_1.get_shape())
            incpt5_1 = slim.conv2d(net, 32, [5, 5], stride=1, scope='incpt5_1')
            print(incpt5_1.get_shape())
            incpt_out = tf.concat([incpt1, incpt3_1, incpt5_1], axis=-1)
            print(incpt_out.get_shape())
    return incpt_out


def incept_decoder(net, is_training=True, reuse=tf.AUTO_REUSE, scope='id'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], 
                            activation_fn=tf.nn.relu, 
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            incpt5_1d = slim.conv2d_transpose(net[:,:,:,96:128], 128, [5, 5], scope='incpt5_1d')  
            print(incpt5_1d.get_shape())
            incpt3_1d = slim.conv2d_transpose(net[:,:,:,33:96], 128, [3, 3], scope='incpt3_1d')
            print(incpt3_1d.get_shape())
            incpt1d = slim.conv2d_transpose(net[:,:,:,0:32], 128, [1, 1], scope='incpt1d') 
            print(incpt1d.get_shape())
            incpt_outd = incpt1d + incpt3_1d + incpt5_1d
            print(incpt_outd.get_shape())
            net = slim.conv2d_transpose(incpt_outd, 64, [7, 7], stride=2, scope='dconv1')
            print(net.get_shape())
            net = slim.conv2d_transpose(net, 1, [7, 7], stride=2, scope='dconv2') # change fm count to 3 if using rgb images
            print(net.get_shape())
    return net


def build_model(origin, is_training, lr):
    incpt_out = incept_encoder(origin, is_training)
    reconst = incept_decoder(incpt_out, is_training)
    if is_training:
        loss=tf.reduce_mean(tf.squared_difference(origin, reconst))
        opt=tf.train.RMSPropOptimizer(lr).minimize(loss)
    
        return loss, opt, reconst
    else:
        return incpt_out


