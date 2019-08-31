from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network import biggandeep
import tensorflow as tf
from dataset.dataset import get_dataset
from training_loop.training_utils import *


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    dataset = get_dataset(name=config.dataset, data_dir=config.data_dir, seed=config.seed)

    with tf.device('/cpu:0'):
        print("Constructing networks...")
        Network = biggandeep.Network(dataset=dataset, model_dir=config.model_dir, run_dir=config.run_dir)
        data_iter = Network.input_data_as_iter(
            batch_size=config.batch_size // config.gpu_nums, seed=config.seed, mode="train")

        eval_iter = Network.input_data_as_iter(
            batch_size=config.batch_size // config.gpu_nums, seed=config.seed, mode="train")
        global_step = tf.compat.v1.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
    print("Building Tensorflow graph...")
    g_grad_pool = []
    d_grad_pool = []
    inception_score_pool = []
    fid_pool = []
    for gpu in range(config.gpu_nums):
        with tf.name_scope("GPU%d" % gpu), tf.device('/gpu:%d' % gpu):
            fs, ls = data_iter.get_next()
            fs, ls = Network.generate_samples(fs, ls)
            g_loss, d_loss = Network.create_loss(fs, ls)
            g_op = Network.get_gen_optimizer()
            d_op = Network.get_disc_optimizer()
            with tf.control_dependencies([g_loss]):
                g_grad_pool.append(g_op.compute_gradients(g_loss, Network.generator.trainable_variables))
            with tf.control_dependencies([d_loss]):
                d_grad_pool.append(d_op.compute_gradients(d_loss, Network.discriminator.trainable_variables))
            f_eval, l_eval = eval_iter.get_next()
            inception_score_shadow, fid_shadow = Network.eval(f_eval, l_eval)
            inception_score_pool.append(inception_score_shadow)
            fid_pool.append(fid_shadow)
    with tf.device('/cpu:0'):
        g_update_op = Network.update(g_grad_pool, g_op, global_step)
        d_update_op = Network.update(d_grad_pool, d_op)
        g_ma_op = Network.ma_op(global_step=global_step)
        merge_op = Network.summary()
        inception_score = tf.reduce_mean(inception_score_pool, 0)
        fid = tf.reduce_mean(fid_pool, 0)

    saver = tf.train.Saver()
    print('Start training...\n')
    # with tf.Session(config=tf.ConfigProto(
    #         allow_soft_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([data_iter.initializer, eval_iter.initializer])
        fsnap, lsnap = sess.run(data_iter.get_next())
        fakes, _ = Network.generate_samples(fsnap, lsnap, is_training=False)
        save_image_grid(fsnap["images"], filename=config.model_dir + '/reals.png')
        summary_writer = tf.summary.FileWriter(logdir=config.model_dir, graph=sess.graph)
        for step in range(config.total_step):
            for D_repeat in range(Network.disc_iters):
                sess.run([d_update_op, g_ma_op])
            sess.run([g_update_op])
            if step % config.summary_per_steps == 0:
                summary_file = sess.run(merge_op)
                summary_writer.add_summary(summary_file, step)
            if step % config.eval_per_steps == config.eval_per_steps // 2:
                timer.update()
                fakes_np = sess.run(fakes["generated"])
                save_image_grid(fakes_np, filename=config.model_dir + '/fakes%06d.png' % step)
                [inception_score_eval, fid_eval] = sess.run([inception_score, fid])
                print("Time %s, fid %f, inception_score %f ,step %d" %
                      (timer.runing_time, fid_eval, inception_score_eval, step))
            if step % config.save_per_steps == 0:
                saver.save(sess, save_path=config.model_dir + '/model.ckpt', global_step=step)
