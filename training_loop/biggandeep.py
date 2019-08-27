from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network import biggandeep
import tensorflow as tf
from dataset.dataset import get_dataset
import time


class Timer(object):
    def __init__(self):
        self._init_time = time.time()
        self._last_update_time = self._init_time
        self._duration = 0

    def update(self):
        cur = time.time()
        self._duration = cur - self._last_update_time
        self._last_update_time = cur

    @property
    def duration(self):
        return self._duration

    @property
    def runing_time(self):
        return self._last_update_time - self._init_time


def training_loop(params=None, steps=250000, model_dir=None):
    timer = Timer()
    dataset = get_dataset(name="imagenet128")
    saver = tf.train.Saver()

    with tf.device('/cpu:0'):
        print("Constructing networks...")
        Network = biggandeep.Network(dataset=dataset, model_dir=model_dir)
        data_iter = Network.input_data_as_iter(params={"batch_size": 32, "seed": 527}, mode="train")
        eval_iter = Network.input_data_as_iter(params={"batch_size": 32, "seed": 527}, mode="eval")
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
    print("Building Tensorflow graph...")
    g_grad_pool = []
    d_grad_pool = []
    for gpu in range(4):
        with tf.name_scope("GPU{}".format(gpu)), tf.device('/gpu:{}'.format(gpu)):
            fs, ls = data_iter.get_next()
            fs, ls = Network.generate_samples(fs, ls)
            g_loss, d_loss = Network.create_loss(fs, ls)
            g_op = Network.get_gen_optimizer()
            d_op = Network.get_disc_optimizer()
            with tf.control_dependencies([g_loss]):
                g_grad_pool.append(g_op.compute_gradients(g_loss, Network.generator.trainable_variables))
            with tf.control_dependencies([d_loss]):
                d_grad_pool.append(d_op.compute_gradients(d_loss, Network.discriminator.trainable_variables))
    with tf.device('/cpu:0'):
        g_update_op = Network.update(g_grad_pool, g_op)
        d_update_op = Network.update(d_grad_pool, d_op)
        g_ma_op = Network.ma_op(global_step=global_step)
        merge_op = Network.summary()
        f_eval, l_eval = eval_iter.get_next()
        [inception_score, fid] = Network.eval(f_eval, l_eval)

    print('Start training...\n')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([data_iter.initializer, eval_iter.initializer])
        summary_writer = tf.summary.FileWriter(logdir=model_dir, graph=sess.graph)
        for step in range(steps):
            for D_repeat in range(Network.disc_iters):
                sess.run([d_update_op, g_ma_op])
            sess.run([g_update_op])
            if step % 100 == 0:
                summary_file = sess.run([merge_op])
                summary_writer.add_summary(summary_file, step)
            if step % 2500 == 0:
                timer.update()
                [inception_score, fid] = sess.run([inception_score, fid])
                print("Time %s, fid %f, inception_score %f ,step %d", timer.runing_time, fid, inception_score, step)
            if step % 10000 == 0:
                saver.restore(sess, save_path=model_dir + 'step{}'.format(step) + "model.ckpt")
