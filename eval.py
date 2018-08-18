from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from config import Config as conf
from model import CapsNet
from utils import load_mnist, save_images


def main(_):
    # Load Graph
    capsNet = CapsNet(is_training=False, routing=2)
    print('[+] Graph is constructed')

    # Load test data
    teX, teY = load_mnist(conf.dataset, is_training=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    # use GPU0
    config.gpu_options.visible_device_list = '3'
    # allocate 50% of GPU memory
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # Start session
    with capsNet.graph.as_default():
        sv = tf.train.Supervisor(logdir=conf.logdir)
        with sv.managed_session(config=config) as sess:
            # Restore parameters
            checkpoint_path = tf.train.latest_checkpoint(conf.logdir)
            sv.saver.restore(sess, checkpoint_path)
            print('[+] Graph is restored from ' + checkpoint_path)

            # Make results directory
            if not os.path.exists('results'):
                os.mkdir('results')

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


if __name__ == "__main__":
    tf.app.run()
