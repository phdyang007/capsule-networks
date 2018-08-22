import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import Config as conf
from utils import *


class CapsNet:
    def __init__(self, is_training=False, routing = 0, dcap_init = 0):
        self.graph = tf.Graph()
        self.is_training=is_training

        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
            self.input = tf.reshape(self.x, [conf.batch_size, 28, 28, 1])
            assert self.input.get_shape() == (conf.batch_size, 28, 28, 1)
            if dcap_init == 0:
                self.build_model(is_training, routing)
            else:
                #build two models: (1) CNN based decap inference model  and (2) caps with bp routing
                self.build_model_2(routing = routing)

    def build_model_dcap_init(self, is_training=False, reuse=False):
        with tf.variable_scope("dcap_init") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                        biases_initializer=tf.constant_initializer(0),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
                net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='dcap_conv1') #28x28x64
                #print net.get_shape()
                net = slim.conv2d(net, 64, [3, 3], stride=2, scope='dcap_pool1') #14x14x64
                #print net.get_shape()
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='dcap_conv2') #14x14x128
                #print net.get_shape()
                net = slim.conv2d(net, 128, [3, 3], stride=2, scope='dcap_pool2') #7x7x128
                #print net.get_shape()
                net = slim.conv2d(net, 10, [3, 3], stride=2, scope='dcap_conv3') #4x4x10
                #print net.get_shape()
                net = tf.reshape(net, [conf.batch_size, 16, 10]) # batch x 16 x 10
                dcap = tf.transpose(net, perm=[0, 2, 1]) # batch x 10 x 16
                assert dcap.get_shape() == (conf.batch_size, 10, 16)

                logits = tf.sqrt(tf.reduce_sum(tf.square(dcap), axis=2) + conf.eps)  # [batch_size, 10]
                probs = tf.nn.softmax(logits)
                preds = tf.argmax(probs, axis=1)
                mloss = self.margin_loss(logits=logits, labels=self.y)
        return mloss, dcap
    
    def build_model_capnets(self, is_training=False):
        with tf.variable_scope("cap_nets") as scope:
            with tf.variable_scope('conv1_layer'):
                conv1 = tf.layers.conv2d(
                    inputs=self.input,
                    filters=256,
                    kernel_size=9,
                    activation=tf.nn.relu,
                )
                assert conv1.get_shape() == (conf.batch_size, 20, 20, 256)

            # Primary Caps
            with tf.variable_scope('primary_caps'):
                pcaps = self.primary_caps(
                    inputs=conv1
                )
                assert pcaps.get_shape() == (conf.batch_size, 32*6*6, 8)

            # Digit Caps
            with tf.variable_scope('digit_caps'):
                dcap = self.digit_caps(
                    inputs=pcaps,
                    num_iters=3,
                    num_caps=10,
                    routing =1
                )
                assert dcap.get_shape() == (conf.batch_size, 10, 16)
        return dcap
    def build_model_2(self, routing = 1):
        self.mloss, dcap = self.build_model_dcap_init(is_training=True)
        _, self.dcap_gt = self.build_model_dcap_init(is_training=False, reuse= True)
        dcap = self.build_model_capnets(is_training=True)
        if routing == 1:
            self.dloss = tf.nn.l2_loss(dcap-self.dcap_gt)
        self.trainable_variables = tf.trainable_variables()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.cnn_vars = [var for var in self.trainable_variables if 'dcap_init' in var.name]
        self.cap_vars = [var for var in self.trainable_variables if 'cap_nets' in var.name]
        self.cnn_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.mloss, global_step=self.global_step, var_list=self.cnn_vars)
        self.cap_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.dloss, global_step=self.global_step, var_list=self.cap_vars) 

    def build_model(self, is_training=False, routing = 0):
        # Input Layer, Reshape to [batch, height, width, channel]


        # ReLU Conv1
        with tf.variable_scope('conv1_layer'):
            conv1 = tf.layers.conv2d(
                inputs=self.input,
                filters=256,
                kernel_size=9,
                activation=tf.nn.relu,
            )
            assert conv1.get_shape() == (conf.batch_size, 20, 20, 256)

        # Primary Caps
        with tf.variable_scope('primary_caps'):
            pcaps = self.primary_caps(
                inputs=conv1
            )
            assert pcaps.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Digit Caps
        with tf.variable_scope('digit_caps'):
            dcaps, self.routing_loss = self.digit_caps(
                inputs=pcaps,
                num_iters=3,
                num_caps=10,
                routing =routing,
            )
            assert dcaps.get_shape() == (conf.batch_size, 10, 16)

        # Prediction
        with tf.variable_scope('prediction'):
            self.logits = tf.sqrt(tf.reduce_sum(tf.square(dcaps), axis=2) + conf.eps)  # [batch_size, 10]
            self.probs = tf.nn.softmax(self.logits)
            self.preds = tf.argmax(self.probs, axis=1)  # [batch_size]
            assert self.logits.get_shape() == (conf.batch_size, 10)

        # Reconstruction
        with tf.variable_scope('reconstruction'):
            targets = tf.argmax(self.y, axis=1) if is_training else self.preds
            self.decoded = self.reconstruction(
                inputs=dcaps,
                targets=targets,
            )
            assert self.decoded.get_shape() == (conf.batch_size, 28, 28)

        if not is_training: return

        # Margin Loss
        with tf.variable_scope('margin_loss'):
            self.mloss = self.margin_loss(
                logits=self.logits,
                labels=self.y,
            )



        # Train
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #self.loss = self.mloss + 0.0005 * self.rloss
        self.loss = self.mloss 

        self.train_vars = [(v.name, v.shape) for v in tf.trainable_variables()]
        self.cnn_vars = [var for var in tf.trainable_variables() if not 'bij' in var.name]
        self.rt_vars = [var for var in tf.trainable_variables() if 'bij' in var.name]
        self.train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss, var_list=self.cnn_vars, global_step=self.global_step)
        self.routing_op =tf.train.AdamOptimizer(conf.learning_rate).minimize(self.routing_loss, var_list=self.rt_vars, global_step=self.global_step)

        # Summary
        tf.summary.scalar('margin_loss', self.mloss)
        tf.summary.scalar('total_loss', self.loss)
        self.summary = tf.summary.merge_all()

        return

    def primary_caps(self, inputs):
        # Validate inputs
        assert inputs.get_shape() == (conf.batch_size, 20, 20, 256)

        # Convolution
#        convs = []
#        for i in range(32):
#            conv = tf.layers.conv2d(
#                inputs=inputs,
#                filters=8,
#                kernel_size=9,
#                strides=2,
#            )
#            assert conv.get_shape() == (conf.batch_size, 6, 6, 8)
#            flat_shape = (conf.batch_size, 6*6, 8)
#            conv_flatten = tf.reshape(conv, flat_shape)
#            convs.append(conv_flatten)
#        convs = tf.concat(convs, axis=1)
#        assert convs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Convolution (batched)
        convs = tf.layers.conv2d(
            inputs=inputs,
            filters=32*8,
            kernel_size=9,
            strides=2,
            activation=tf.nn.relu,
        )
        convs = tf.reshape(convs, [conf.batch_size, -1, 8])
        assert convs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Squash
        pcaps = self.squash(convs)
        return pcaps

    def digit_caps(self, inputs, num_iters, num_caps, routing=0):
        #routing methods: 0: dynamic routing, 1 for bp routing
        # Validate inputs
        assert inputs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Reshape input
        u = tf.reshape(inputs, [conf.batch_size, 32*6*6, 1, 8, 1])
        u = tf.tile(u, [1, 1, num_caps, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 1]
        
        # Dynamic routing
        bij = tf.zeros((32*6*6, num_caps), name='b')
        wij = tf.get_variable('wij', shape=(1, 32*6*6, num_caps, 8, 16), initializer=tf.contrib.layers.xavier_initializer())
        w = tf.tile(wij, [conf.batch_size, 1, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 16]
        # bij = tf.tile(tf.zeros((32, num_caps), name='b'), [6*6, 1])  # [32*6*6, num_caps]
        # wij = tf.get_variable('wij', shape=(1, 32, num_caps, 8, 16))
        # w = tf.tile(wij, [conf.batch_size, 6*6, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 16]

        # uhat
        uhat = tf.matmul(u, w, transpose_a=True)  # [batch_size, 32*6*6, num_caps, 1, 16]
        uhat = tf.reshape(uhat, [conf.batch_size, 32*6*6, num_caps, 16])  # [batch_size, 32*6*6, num_caps, 16]
        if routing == 0:
            uhat_stop_grad = tf.stop_gradient(uhat)
            assert uhat.get_shape() == (conf.batch_size, 32*6*6, num_caps, 16)

            for r in range(num_iters):
                with tf.variable_scope('routing_iter_' + str(r)):
                    # cij
                    cij = tf.nn.softmax(bij, dim=-1)  # [32*6*6, num_caps]
                    cij = tf.tile(tf.reshape(cij, [1, 32*6*6, num_caps, 1]),
                                [conf.batch_size, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 1]
                    assert cij.get_shape() == (conf.batch_size, 32*6*6, num_caps, 1)

                    if r == num_iters-1: 
                        # s, v
                        s = tf.reduce_sum(tf.multiply(uhat, cij), axis=1)  # [batch_size, num_caps, 16]
                        v = self.squash(s)  # [batch_size, num_caps, 16]
                        assert v.get_shape() == (conf.batch_size, num_caps, 16)
                    else: 
                        # s, v (with no gradient propagation)
                        s = tf.reduce_sum(tf.multiply(uhat_stop_grad, cij), axis=1)  # [batch_size, num_caps, 16]
                        v = self.squash(s)  # [batch_size, num_caps, 16]
                        assert v.get_shape() == (conf.batch_size, num_caps, 16)
                        
                        # update b
                        vr = tf.reshape(v, [conf.batch_size, 1, num_caps, 16])
                        a = tf.reduce_sum(tf.reduce_sum(tf.multiply(uhat_stop_grad, vr), axis=0), axis=2)  # [32*6*6, num_caps]
                        bij = bij + a
                        assert a.get_shape() == (32*6*6, num_caps)
        if routing == 1:
            bij = tf.get_variable('bij', shape=(32*6*6, num_caps), initializer=tf.constant_initializer(0.0))
            cij = tf.nn.softmax(bij)
            cij = tf.tile(tf.reshape(cij, [1, 32*6*6, num_caps, 1]),
                        [conf.batch_size, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 1]
            s = tf.reduce_sum(tf.multiply(uhat, cij), axis=1)  # [batch_size, num_caps, 16]
            v = self.squash(s)  # [batch_size, num_caps, 16]
        if routing == 2:
            bij = tf.get_variable('bij', shape=(32*6*6, num_caps), initializer=tf.constant_initializer(1.0))
            #bij = tf.get_variable('bij', shape=(32*6*6, num_caps), initializer=tf.contrib.layers.xavier_initializer())
            #cij = tf.nn.softmax(bij, axis=0) 
            cij=bij

            if self.is_training:
                label = 2*self.y-1.0

                cij_trans=tf.transpose(cij)  #10 x 3266
                cij_lens=tf.reduce_mean(tf.square(cij_trans), axis=1) #10 , 
                #cij_regularization=tf.reduce_sum(tf.multiply(label, tf.tile(tf.reshape(cij_lens, [1, 10]), [conf.batch_size, 1])), axis=1) #batch x 10
                cij_regularization=tf.reduce_sum(tf.tile(tf.reshape(cij_lens, [1, 10]), [conf.batch_size, 1]), axis=1) #batch x 10
            cij = tf.tile(tf.reshape(cij, [1, 32*6*6, num_caps, 1]),
                        [conf.batch_size, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 1]
            s = tf.reduce_sum(tf.multiply(uhat, cij), axis=1)  # [batch_size, num_caps, 16]
            v = self.squash(s)  # [batch_size, num_caps, 16]
            if self.is_training:
                v_len = tf.multiply(tf.reduce_sum(tf.square(v), axis=-1),label)  #batch num_caps
                v_loss= tf.reduce_sum(v_len, axis=-1)
                c_loss= cij_regularization
                return v, tf.reduce_mean(0.00001*c_loss-v_loss)
            else:
                return v, 0

    def squash(self, s):
        s_l2 = tf.sqrt(tf.reduce_sum(tf.square(s), axis=-1, keep_dims=True) + conf.eps)
        scalar_factor = tf.square(s_l2) / (1 + tf.square(s_l2))
        v = scalar_factor * tf.divide(s, s_l2)
        return v

    def reconstruction(self, inputs, targets):
        # Validation
        assert inputs.get_shape() == (conf.batch_size, 10, 16)
        assert targets.get_shape() == (conf.batch_size)

        with tf.variable_scope('masking'):
            enum = tf.cast(tf.range(conf.batch_size), dtype=tf.int64)
            enum_indices = tf.concat(
                [tf.expand_dims(enum, 1), tf.expand_dims(targets, 1)],
                axis=1
            )
            assert enum_indices.get_shape() == (conf.batch_size, 2)

            masked_inputs = tf.gather_nd(inputs, enum_indices)
            assert masked_inputs.get_shape() == (conf.batch_size, 16)

        with tf.variable_scope('reconstruction'):
            fc_relu1 = tf.contrib.layers.fully_connected(
                inputs=masked_inputs,
                num_outputs=512,
                activation_fn=tf.nn.relu
            )
            # fc_relu1 = tf.nn.dropout(fc_relu1, keep_prob=0.9)
            fc_relu2 = tf.contrib.layers.fully_connected(
                inputs=fc_relu1,
                num_outputs=1024,
                activation_fn=tf.nn.relu
            )
            # fc_relu2 = tf.nn.dropout(fc_relu2, keep_prob=0.9)
            fc_sigmoid = tf.contrib.layers.fully_connected(
                inputs=fc_relu2,
                num_outputs=784,
                activation_fn=tf.nn.sigmoid
            )
            assert fc_sigmoid.get_shape() == (conf.batch_size, 784)
            recons = tf.reshape(fc_sigmoid, shape=(conf.batch_size, 28, 28))

        return recons

    def margin_loss(self, logits, labels, mplus=0.9, mminus=0.1, lambd=0.5):
        left = tf.square(tf.maximum(0., mplus - logits))
        right = tf.square(tf.maximum(0., logits - mminus))
        assert left.get_shape() == (conf.batch_size, 10)
        assert right.get_shape() == (conf.batch_size, 10)

        T_k = labels
        L_k = T_k * left + lambd * (1-T_k) * right
        mloss = tf.reduce_mean(tf.reduce_sum(L_k, axis=1))
        return mloss

    def reconstruction_loss(self, origin, decoded):
        origin = tf.reshape(origin, shape=(conf.batch_size, -1))
        decoded = tf.reshape(decoded, shape=(conf.batch_size, -1))
        rloss = tf.reduce_mean(tf.square(decoded - origin))
        return rloss

    #def digit_caps_model(self, inputs, targets)
