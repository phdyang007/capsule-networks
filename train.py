from __future__ import print_function

import numpy as np
import tensorflow as tf
from config import Config as conf
from datetime import datetime
from model import CapsNet
import time


def main(_):
    # Construct Graph
    capsNet = CapsNet(is_training=True, routing=0)
    print('[+] Graph is constructed')
    config = tf.ConfigProto()
    # use GPU0
    config.gpu_options.visible_device_list = '0, 2'
    # allocate 50% of GPU memory
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # Start session
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=conf.logdir,
                             save_model_secs=0)
    with sv.managed_session(config=config) as sess:
        start_epoch = 0
        try:
            # checkpoint_path = 'logdir/model_epoch_%02d' % start_epoch
            checkpoint_path = tf.train.latest_checkpoint(conf.logdir)
            print(checkpoint_path)
            sv.saver.restore(sess, checkpoint_path)
        except:
              print('start new model')
        
        print('[+] Trainable variables')
        for tvar in capsNet.train_vars: print(tvar)

        print('caps var')
        print(capsNet.caps_vars)
        print('reconstruction_vars')
        print(capsNet.reconstruction_vars)
        print('routing_vars')
        print(capsNet.routing_vars)

        print('[+] Training start')
        for epoch in range(conf.num_epochs):
            start_time = time.time()
            if sv.should_stop(): break
            losses = []
            accuracy = []

            # from tqdm import tqdm
            # for step in tqdm(range(capsNet.num_batch), total=capsNet.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(capsNet.num_batch):
                # _, loss, rloss, mloss = sess.run([capsNet.train_op, capsNet.loss, capsNet.rloss, capsNet.mloss])
                _, _, loss, rloss, mloss, acc = sess.run([capsNet.train_caps_op,
capsNet.train_reconstruction_op, capsNet.loss, capsNet.rloss, capsNet.mloss,
capsNet.accuracy
])
                losses.append(loss)
                accuracy.append(acc)
                if step % 100 == 0:
                    print("%s: epoch[%g/%g], step[%g/%g], loss[%f], rloss[%f], mloss[%f], acc[%.2f]"%(datetime.now(), epoch, conf.num_epochs, step,
capsNet.num_batch, loss, rloss, mloss, acc
))
            current_time = time.time()
            print('[+] %.2f EPOCH %d : acc: %.2f' % (current_time-start_time, epoch,
np.mean(accuracy)) + str(np.mean(losses)))
            gs = sess.run(capsNet.global_step)
            sv.saver.save(sess, conf.logdir + '/model_epoch_%02d' %
(epoch+start_epoch+1))

    print("[+] Training is Completed")
    return

if __name__ == "__main__":
    tf.app.run()
