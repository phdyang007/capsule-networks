from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from config import Config as conf
from model import CapsNet
from utils import load_mnist, save_images


def main(_):
    # Load Graph
    capsNet = CapsNet(is_training=False, routing=0)
    print('[+] Graph is constructed')

    # Load test data
    teX, teY = load_mnist(conf.dataset, is_training=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    # use GPU0
    config.gpu_options.visible_device_list = '0, 1, 2'
    # allocate 50% of GPU memory
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # Start session
    with capsNet.graph.as_default():
        sv = tf.train.Supervisor(logdir=conf.logdir)
        with sv.managed_session(config=config) as sess:

            lodir_list = ['logdir/model_epoch_%02d'%n for n in range(0, 50)]
            print(lodir_list)
            fo = open('result_acc.txt', "w")
            for idx, lodir in enumerate(lodir_list):
                 try:
                     # Restore parameters
                     checkpoint_path = tf.train.latest_checkpoint(conf.logdir)
                     print(checkpoint_path)
                     sv.saver.restore(sess, lodir)
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
                         recon_imgs = sess.run(capsNet.decoded, {capsNet.x: teX[start:end]})
                         recon_imgs = np.reshape(recon_imgs, (conf.batch_size, -1))
                         orgin_imgs = np.reshape(teX[start:end], (conf.batch_size, -1))
                         squared = np.square(recon_imgs - orgin_imgs)
                         reconstruction_err.append(np.mean(squared))
                         if i % 5 == 0:
                             imgs = np.reshape(recon_imgs, (conf.batch_size, 28, 28, 1))
                             size = 6
                             save_images(imgs[0:size * size, :], [size, size], 'results/test_%03d.png' % i)
              
                         # Classification
                         cls_result = sess.run(capsNet.preds, {capsNet.x: teX[start:end]})
                         cls_answer = teY[start:end]
                         cls_acc = np.mean(np.equal(cls_result, cls_answer).astype(np.float32))
                         classification_acc.append(cls_acc)
                     # Print classification accuracy & reconstruction error
                     print('reconstruction_err : ' + str(np.mean(reconstruction_err)))
                     print('classification_acc : ' + str(np.mean(classification_acc) * 100))
                     fo.write("%d: %.2f\n" %(idx, np.mean(classification_acc)*100))
                 except:
                     print('fail to load model')
             

            fo.close()


if __name__ == "__main__":
    tf.app.run()
