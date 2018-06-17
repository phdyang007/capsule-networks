from __future__ import print_function

import numpy as np
import tensorflow as tf
from config import Config as conf
from datetime import datetime
from model import CapsNet

def main(_):
    # Construct Graph
    capsNet = CapsNet(is_training=True, routing=1, dcap_init=1)
    print('[+] Graph is constructed')
    quit()
    config = tf.ConfigProto()
    # use GPU0
    config.gpu_options.visible_device_list = '2'
    # allocate 50% of GPU memory
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # Start session
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=conf.logdir,
                             save_model_secs=0)
    with sv.managed_session(config=config) as sess:
        print('[+] Trainable variables')
        for tvar in capsNet.train_vars: print(tvar)

        print('[+] Training start')
        for epoch in range(conf.num_epochs):
            if sv.should_stop(): break
            losses = []

            # from tqdm import tqdm
            # for step in tqdm(range(capsNet.num_batch), total=capsNet.num_batch, ncols=70, leave=False, unit='b'):
            for step in range(capsNet.num_batch):
                _, loss, rloss, mloss = sess.run([capsNet.train_op, capsNet.loss, capsNet.rloss, capsNet.mloss])
                losses.append(loss)
                if step % 100 == 0:
                    print("%s: epoch[%g/%g], step[%g/%g], loss[%f], rloss[%f], mloss[%f]"%(datetime.now(), epoch, conf.num_epochs, step, capsNet.num_batch, loss, rloss, mloss))
            print(('[+] EPOCH %d : ' % epoch) + str(np.mean(losses)))
            gs = sess.run(capsNet.global_step)
            sv.saver.save(sess, conf.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("[+] Training is Completed")
    return

if __name__ == "__main__":
    tf.app.run()