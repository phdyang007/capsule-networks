from __future__ import print_function

import numpy as np
import tensorflow as tf
from config import Config as conf
from datetime import datetime
from model import CapsNet
from utils import load_mnist, save_images

def main(_):
    # Construct Graph
    capsNet = CapsNet(is_training=True, routing=2)
    print('[+] Graph is constructed')
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
                _,_, loss, rloss, mloss = sess.run([capsNet.train_op, capsNet.routing_op, capsNet.loss, capsNet.routing_loss, capsNet.mloss])
                losses.append(loss)
                if step % 100 == 0:
                    print("%s: epoch[%g/%g], step[%g/%g], loss[%f], rloss[%f], mloss[%f]"%(datetime.now(), epoch, conf.num_epochs, step, capsNet.num_batch, loss, rloss, mloss))
            print(('[+] EPOCH %d : ' % epoch) + str(np.mean(losses)))
            gs = sess.run(capsNet.global_step)
            sv.saver.save(sess, conf.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

            ######
            teX, teY = load_mnist(conf.dataset, is_training=False)
            # Start session

            reconstruction_err = []
            classification_acc = []
            for i in range(10000 // conf.batch_size):
                start = i * conf.batch_size
                end = start + conf.batch_size

                # Reconstruction


                # Classification
                cls_result = sess.run(capsNet.preds, {capsNet.x: teX[start:end]})
                cls_answer = teY[start:end]
                cls_acc = np.mean(np.equal(cls_result, cls_answer).astype(np.float32))
                classification_acc.append(cls_acc)

            # Print classification accuracy & reconstruction error
            print('classification_acc : ' + str(np.mean(classification_acc) * 100))
            ######

    print("[+] Training is Completed")
    return

if __name__ == "__main__":
    tf.app.run()
