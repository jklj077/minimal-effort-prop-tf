import os
import itertools
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from meprop import *

tf.app.flags.DEFINE_string("data_dir", "mnist/input_data",
                           "dir for storing mnist data")
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate")
tf.app.flags.DEFINE_string("dev", "/cpu:0", "device")

FLAGS = tf.app.flags.FLAGS


class Mnist(object):
    def __init__(self,
                 nlayer,
                 nhidden,
                 nminibatch,
                 dropout,
                 meprop_k=0,
                 meprop_r=None):
        self.data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        self.nminibatch = nminibatch
        self.dropout = dropout
        self.graph = tf.Graph()

        with self.graph.as_default():
            with self.graph.device(FLAGS.dev):
                with tf.name_scope('input'):
                    self.x = tf.placeholder(
                        tf.float32, [None, 784], name='x-input')
                    self.y_ = tf.placeholder(
                        tf.float32, [None, 10], name='y-input')

                self.keep_prob = tf.placeholder(tf.float32)
                hiddens = []
                meprop = (meprop_k > 0)

                if nlayer == 1:
                    hidden = self.nn_layer(
                        self.x, 784, 10, 'layer', act=tf.identity)
                    hiddens.append(hidden)
                else:
                    for i in range(0, nlayer):
                        if i == 0:
                            hidden = self.nn_layer(
                                self.x,
                                784,
                                nhidden,
                                'layer' + str(i),
                                meprop=meprop,
                                meprop_k=meprop_k,
                                meprop_r=meprop_r)
                            dropped = tf.nn.dropout(hidden, self.keep_prob)
                            hiddens.append(dropped)
                        elif i == nlayer - 1:
                            hidden = self.nn_layer(
                                hiddens[-1],
                                nhidden,
                                10,
                                'layer' + str(i),
                                act=tf.identity)
                            hiddens.append(hidden)
                        else:
                            hidden = self.nn_layer(
                                hiddens[-1],
                                nhidden,
                                nhidden,
                                'layer' + str(i),
                                meprop=meprop,
                                meprop_k=meprop_k,
                                meprop_r=meprop_r)
                            dropped = tf.nn.dropout(hidden, self.keep_prob)
                            hiddens.append(dropped)

                y = hiddens[-1]

                with tf.name_scope('cross_entropy'):
                    diff = tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.y_, logits=y)
                    with tf.name_scope('total'):
                        cross_entropy = tf.reduce_mean(diff)

                with tf.name_scope('train'):
                    self.train_step = tf.train.AdamOptimizer(
                        FLAGS.learning_rate).minimize(cross_entropy)

                with tf.name_scope('accuracy'):
                    with tf.name_scope('correct_prediction'):
                        correct_prediction = tf.equal(
                            tf.argmax(y, 1), tf.argmax(self.y_, 1))
                    with tf.name_scope('accuracy'):
                        self.accuracy = tf.reduce_mean(
                            tf.cast(correct_prediction, tf.float32))

                self.initializer = tf.global_variables_initializer()
        self.graph.finalize()

    def __call__(self, nepoch, flog):
        best = 0.0
        besti = 0
        bestt = 0.0
        ttime = 0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=self.graph, config=config) as sess:
            sess.run(self.initializer)
            for e in range(nepoch):
                for i in range(int(55000 / self.nminibatch)):
                    start = time.time()
                    sess.run(
                        [self.train_step],
                        feed_dict=self.feed_dict(True, False))
                    ttime += time.time() - start
                    if (i * self.nminibatch) % 5000 == 0:  # dev-set accuracy
                        acc = sess.run(
                            [self.accuracy],
                            feed_dict=self.feed_dict(False, True))
                        if acc[0] > best:
                            best = acc[0]
                            besti = e * 11 + int((i * self.nminibatch) / 5000)
                            tacc = sess.run(
                                [self.accuracy],
                                feed_dict=self.feed_dict(False, False))
                            bestt = tacc[0]
                            print('Accuracy at {0}:{1} {2:f}:{3:f}'.format(
                                e, int(i * self.nminibatch / 5000), acc[0],
                                tacc[0]))
                        else:
                            print('Accuracy at {0}:{1} {2:f}'.format(
                                e, int(i * self.nminibatch / 5000), acc[0]))
                flog.write(
                    'till:{0}, best {1} at {2} with {3}, time {4:f}\n'.format(
                        e, best, besti, bestt, ttime / (e + 1)))
                flog.flush()
            flog.write('at {}, {:.2f}|{:.2f}\n'.format(besti, best * 100,
                                                       bestt * 100))
            flog.flush()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self,
                 input_tensor,
                 input_dim,
                 output_dim,
                 layer_name,
                 act=tf.nn.relu,
                 meprop=False,
                 meprop_k=0,
                 meprop_r=None):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b'):
                if meprop:
                    preactivate = meprop_r(input_tensor, weights,
                                           meprop_k) + biases
                else:
                    preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
            return activations

    def feed_dict(self, train, dev):
        if train:
            xs, ys = self.data.train.next_batch(self.nminibatch)
            k = self.dropout
        elif dev:
            xs, ys = self.data.validation.images, self.data.validation.labels
            k = 1.0
        else:
            xs, ys = self.data.test.images, self.data.test.labels
            k = 1.0
        return {self.x: xs, self.y_: ys, self.keep_prob: k}


def write(flog, s):
    print(s.strip())
    flog.write(s)
    flog.flush()


def main(_):
    dropouts = [0.9]
    layers = [6, 5, 4, 3, 2]
    minibatches = [100]
    hiddens = [25]
    ks = [0]
    runs = range(0, 3)
    meprop_r = [
        MatMulMeProp, MatMulMePropSparse, MatMulMePropSparseOne,
        MatMulMePropUnified, MatMulMePropUnifiedCompacted
    ]

    with open('log.' + str(int(time.time())) + '.txt', 'w') as flog:
        for run, layer, minibatch, hsize, dropout, k in itertools.product(
                runs, layers, minibatches, hiddens, dropouts, ks):
            if k == 0:
                write(
                    flog,
                    'run:{}, layer:{}, minibatch:{}, hidden:{} dropout:{}, k:{}\n'.
                    format(run, layer, minibatch, hsize, dropout, k))
                instance = Mnist(layer, hsize, minibatch, dropout)
                instance(FLAGS.epoch, flog)
                write(flog, '\n')
            else:
                for rou in meprop_r:
                    write(
                        flog,
                        'run:{}, layer:{}, minibatch:{}, hidden:{} dropout:{}, k:{}, rou:{}\n'.
                        format(run, layer, minibatch, hsize, dropout, k,
                               str(rou)))
                    instance = Mnist(layer, hsize, minibatch, dropout, k, rou)
                    instance(FLAGS.epoch, flog)
                    write(flog, '\n')


if __name__ == '__main__':
    tf.app.run(main=main)
