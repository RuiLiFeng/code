from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network import TwoStageBiggandeep
import tensorflow as tf
from dataset.dataset import get_dataset


class Timer(object):
    def __init__(self):
        pass

    def print(self):
        pass


def training_loop(params=None, model_dir=None):
    timer = Timer()
    dataset = get_dataset(name="imagenet128")

    with tf.device('/gpu:0'):
        print("Constructing networks...")
        Network = TwoStageBiggandeep.Network(dataset=dataset, parameters=params, model_dir=None)
        Network.set_training_param(params)
        data_iter = Network.input_data_as_iter(params={"batch_size": 32, "seed": 527}, mode="train")
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
            g_grad_pool.append(g_op.compute_gradients(g_loss, Network.generator.trainable_variables))
            d_grad_pool.append(d_op.compute_gradients(d_loss, Network.discriminator.trainable_variables))
    g_update_op = Network.update(g_grad_pool, g_op)
    d_update_op = Network.update(d_grad_pool, d_op)

    print('Start training...\n')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(Network.step):
            for mb_repeat in range(Network.g_step):
                for D_repeat in range(Network.d_step):
                    sess.run([d_update_op])
                sess.run([g_update_op])  # Still need to enable moving avarage.
            if step % 3000 == 0:
                Network.summary()
                timer.print()
                Network.eval_and_save()





