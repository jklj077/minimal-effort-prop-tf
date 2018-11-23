import os
import itertools
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from meprop import *


tf.app.flags.DEFINE_string("data_dir", "mnist/input_data",
                           "dir for storing mnist data")
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_integer("nlayer", 3, "layer of MLP")
tf.app.flags.DEFINE_integer("minibatch", 32, "minibatch size")
tf.app.flags.DEFINE_integer("hidden_size", 500, "hidden size of MLP")
tf.app.flags.DEFINE_integer("k", 30, "meprop top k")
tf.app.flags.DEFINE_float("keep_rate", 1.0, "keep_rate in dropout")

FLAGS = tf.app.flags.FLAGS

class Mnist(object):
    def __init__(self,
                 nlayer,
                 nhidden,
                 nminibatch,
                 dropout,
                 meprop_k=0,
                 meprop_r=None):
        self.layer = nlayer
        self.hidden_size = nhidden
        self.k = meprop_k

        self.data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        self.nminibatch = nminibatch
        self.dropout = dropout
        self.graph = tf.Graph()

        with self.graph.as_default():
            with self.graph.device('/gpu:0'):
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
                    self.train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

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
        print('layer:{}, minibatch:{}, hidden:{}, keep_rate:{}, k:{}'.
            format(self.layer, self.nminibatch, self.hidden_size, self.dropout, self.k
                    ), file=flog, flush=True)

        best = 0.0
        besti = 0
        bestt = 0.0
        ttime = 0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=self.graph, config=config) as sess:
            sess.run(self.initializer)
            max_steps = (55000*nepoch+self.nminibatch-1)//self.nminibatch
            for i in range(max_steps):
            #for e in range(nepoch):
            #for i in range(int(55000 / self.nminibatch)):
                start = time.time()
                sess.run(
                    [self.train_step],
                    feed_dict=self.feed_dict(True, False))
                ttime += time.time() - start
                if i % 150 == 0:  # dev-set accuracy
                    acc = sess.run(
                        [self.accuracy],
                        feed_dict=self.feed_dict(False, True))
                    if acc[0] > best:
                        best = acc[0]
                        besti = i
                        tacc = sess.run(
                            [self.accuracy],
                            feed_dict=self.feed_dict(False, False))
                        bestt = tacc[0]
                        print('Accuracy at {} {:.2f}:{:.2f}, time {:f}'.format(
                            i, acc[0]*100, tacc[
                                0]*100, ttime/(i+1)), file=flog, flush=True)
                    else:
                        print('Accuracy at {} {:.2f}, time {:f}'.format(i, acc[0]*100, ttime/(i+1)), file=flog, flush=True)
            print('best dev acc {}, with {} at {}'.format(best, bestt, besti), file=flog, flush=True)
            print('', file=flog, flush=True)


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

def main(_):
    with open('log.' + str(int(time.time())) + '.txt', 'w') as flog:
        # print('layer:{}, minibatch:{}, hidden:{}, keep_rate:{}'.
        #     format(FLAGS.nlayer, FLAGS.minibatch, FLAGS.hidden_size, FLAGS.keep_rate), file=flog, flush=True)
        # instance = Mnist(FLAGS.nlayer, FLAGS.hidden_size, FLAGS.minibatch, FLAGS.keep_rate)
        # instance(FLAGS.epoch, flog)
        # print('', file=flog, flush=True)
        
        # print('layer:{}, minibatch:{}, hidden:{}, keep_rate:{}, k:{}'.
        #     format(FLAGS.nlayer, FLAGS.minibatch, FLAGS.hidden_size, FLAGS.keep_rate, FLAGS.k
        #             ), file=flog, flush=True)
        instance = Mnist(FLAGS.nlayer, FLAGS.hidden_size, FLAGS.minibatch, FLAGS.keep_rate, FLAGS.k, MatMulMeProp)
        instance(FLAGS.epoch, flog)
        #print('', file=flog, flush=True)


if __name__ == '__main__':
    tf.app.run(main=main)


# layer:3, minibatch:100, hidden:500, keep_rate:0.9
# Accuracy at 0:0 11.56:12.43
# Accuracy at 0:1 60.34:61.04
# Accuracy at 0:2 75.98:77.30
# Accuracy at 0:3 83.18:83.81
# Accuracy at 0:4 85.78:86.40
# Accuracy at 0:5 88.02:88.16
# Accuracy at 0:6 89.04:89.10
# Accuracy at 0:7 90.68:90.13
# Accuracy at 0:8 90.96:90.47
# Accuracy at 0:9 91.40:91.03
# Accuracy at 0:10 91.66:91.33
# till:0, best dev acc 91.66 at 10 with test acc 91.33, time 2.268853

# Accuracy at 1:0 92.22:91.60
# Accuracy at 1:1 92.40:91.82
# Accuracy at 1:2 92.66:92.25
# Accuracy at 1:3 93.04:92.47
# Accuracy at 1:4 93.30:92.81
# Accuracy at 1:5 93.50:93.03
# Accuracy at 1:6 93.56:93.29
# Accuracy at 1:7 93.88:93.49
# Accuracy at 1:8 93.86
# Accuracy at 1:9 94.00:93.67
# Accuracy at 1:10 93.88
# till:1, best dev acc 94.00 at 20 with test acc 93.67, time 2.237966

# Accuracy at 2:0 94.60:94.11
# Accuracy at 2:1 94.58
# Accuracy at 2:2 94.72:94.23
# Accuracy at 2:3 94.80:94.33
# Accuracy at 2:4 95.02:94.67
# Accuracy at 2:5 95.18:94.67
# Accuracy at 2:6 95.36:94.73
# Accuracy at 2:7 95.40:94.89
# Accuracy at 2:8 95.42:95.02
# Accuracy at 2:9 95.56:95.29
# Accuracy at 2:10 95.60:95.17
# till:2, best dev acc 95.60 at 32 with test acc 95.17, time 2.268341

# Accuracy at 3:0 95.50
# Accuracy at 3:1 95.74:95.37
# Accuracy at 3:2 95.76:95.40
# Accuracy at 3:3 95.82:95.42
# Accuracy at 3:4 95.92:95.59
# Accuracy at 3:5 95.96:95.69
# Accuracy at 3:6 96.16:95.71
# Accuracy at 3:7 96.24:95.66
# Accuracy at 3:8 96.42:95.84
# Accuracy at 3:9 96.04
# Accuracy at 3:10 96.44:95.88
# till:3, best dev acc 96.44 at 43 with test acc 95.88, time 2.251998

# Accuracy at 4:0 96.22
# Accuracy at 4:1 96.54:95.98
# Accuracy at 4:2 96.46
# Accuracy at 4:3 96.30
# Accuracy at 4:4 96.68:96.13
# Accuracy at 4:5 96.56
# Accuracy at 4:6 96.78:96.08
# Accuracy at 4:7 96.46
# Accuracy at 4:8 96.58
# Accuracy at 4:9 96.74
# Accuracy at 4:10 96.78
# till:4, best dev acc 96.78 at 50 with test acc 96.08, time 2.278280

# Accuracy at 5:0 96.76
# Accuracy at 5:1 96.90:96.57
# Accuracy at 5:2 96.74
# Accuracy at 5:3 96.96:96.56
# Accuracy at 5:4 96.68
# Accuracy at 5:5 96.92
# Accuracy at 5:6 96.88
# Accuracy at 5:7 97.00:96.60
# Accuracy at 5:8 96.94
# Accuracy at 5:9 97.06:96.74
# Accuracy at 5:10 97.04
# till:5, best dev acc 97.06 at 64 with test acc 96.74, time 2.246670

# Accuracy at 6:0 96.98
# Accuracy at 6:1 96.86
# Accuracy at 6:2 97.02
# Accuracy at 6:3 96.96
# Accuracy at 6:4 97.02
# Accuracy at 6:5 97.02
# Accuracy at 6:6 97.00
# Accuracy at 6:7 97.10:97.05
# Accuracy at 6:8 97.04
# Accuracy at 6:9 97.14:97.06
# Accuracy at 6:10 97.30:97.04
# till:6, best dev acc 97.30 at 76 with test acc 97.04, time 2.222257

# Accuracy at 7:0 97.22
# Accuracy at 7:1 97.12
# Accuracy at 7:2 97.18
# Accuracy at 7:3 97.32:97.18
# Accuracy at 7:4 97.12
# Accuracy at 7:5 97.28
# Accuracy at 7:6 97.32
# Accuracy at 7:7 97.22
# Accuracy at 7:8 97.28
# Accuracy at 7:9 97.22
# Accuracy at 7:10 97.36:97.13
# till:7, best dev acc 97.36 at 87 with test acc 97.13, time 2.193251

# Accuracy at 8:0 97.38:97.18
# Accuracy at 8:1 97.20
# Accuracy at 8:2 97.26
# Accuracy at 8:3 97.20
# Accuracy at 8:4 97.50:97.26
# Accuracy at 8:5 97.48
# Accuracy at 8:6 97.60:97.26
# Accuracy at 8:7 97.38
# Accuracy at 8:8 97.38
# Accuracy at 8:9 97.30
# Accuracy at 8:10 97.48
# till:8, best dev acc 97.60 at 94 with test acc 97.26, time 2.179903

# Accuracy at 9:0 97.48
# Accuracy at 9:1 97.32
# Accuracy at 9:2 97.28
# Accuracy at 9:3 97.40
# Accuracy at 9:4 97.32
# Accuracy at 9:5 97.50
# Accuracy at 9:6 97.54
# Accuracy at 9:7 97.38
# Accuracy at 9:8 97.44
# Accuracy at 9:9 97.52
# Accuracy at 9:10 97.38
# till:9, best dev acc 97.60 at 94 with test acc 97.26, time 2.170953

# Accuracy at 10:0 97.62:97.38
# Accuracy at 10:1 97.50
# Accuracy at 10:2 97.50
# Accuracy at 10:3 97.52
# Accuracy at 10:4 97.60
# Accuracy at 10:5 97.58
# Accuracy at 10:6 97.72:97.31
# Accuracy at 10:7 97.46
# Accuracy at 10:8 97.48
# Accuracy at 10:9 97.64
# Accuracy at 10:10 97.62
# till:10, best dev acc 97.72 at 116 with test acc 97.31, time 2.146135

# Accuracy at 11:0 97.56
# Accuracy at 11:1 97.62
# Accuracy at 11:2 97.44
# Accuracy at 11:3 97.68
# Accuracy at 11:4 97.68
# Accuracy at 11:5 97.68
# Accuracy at 11:6 97.46
# Accuracy at 11:7 97.56
# Accuracy at 11:8 97.48
# Accuracy at 11:9 97.70
# Accuracy at 11:10 97.64
# till:11, best dev acc 97.72 at 116 with test acc 97.31, time 2.149961

# Accuracy at 12:0 97.68
# Accuracy at 12:1 97.64
# Accuracy at 12:2 97.66
# Accuracy at 12:3 97.58
# Accuracy at 12:4 97.76:97.48
# Accuracy at 12:5 97.68
# Accuracy at 12:6 97.62
# Accuracy at 12:7 97.72
# Accuracy at 12:8 97.64
# Accuracy at 12:9 97.68
# Accuracy at 12:10 97.60
# till:12, best dev acc 97.76 at 136 with test acc 97.48, time 2.154311

# Accuracy at 13:0 97.54
# Accuracy at 13:1 97.74
# Accuracy at 13:2 97.60
# Accuracy at 13:3 97.66
# Accuracy at 13:4 97.64
# Accuracy at 13:5 97.68
# Accuracy at 13:6 97.74
# Accuracy at 13:7 97.46
# Accuracy at 13:8 97.70
# Accuracy at 13:9 97.60
# Accuracy at 13:10 97.84:97.51
# till:13, best dev acc 97.84 at 153 with test acc 97.51, time 2.150240

# Accuracy at 14:0 97.80
# Accuracy at 14:1 97.80
# Accuracy at 14:2 97.66
# Accuracy at 14:3 97.72
# Accuracy at 14:4 97.58
# Accuracy at 14:5 97.88:97.69
# Accuracy at 14:6 97.80
# Accuracy at 14:7 97.68
# Accuracy at 14:8 97.52
# Accuracy at 14:9 97.62
# Accuracy at 14:10 97.70
# till:14, best dev acc 97.88 at 159 with test acc 97.69, time 2.143613

# Accuracy at 15:0 97.76
# Accuracy at 15:1 97.82
# Accuracy at 15:2 97.90:97.77
# Accuracy at 15:3 97.78
# Accuracy at 15:4 97.78
# Accuracy at 15:5 97.68
# Accuracy at 15:6 97.84
# Accuracy at 15:7 97.82
# Accuracy at 15:8 97.56
# Accuracy at 15:9 97.82
# Accuracy at 15:10 97.76
# till:15, best dev acc 97.90 at 167 with test acc 97.77, time 2.142228

# Accuracy at 16:0 97.76
# Accuracy at 16:1 97.80
# Accuracy at 16:2 97.68
# Accuracy at 16:3 97.86
# Accuracy at 16:4 97.88
# Accuracy at 16:5 97.88
# Accuracy at 16:6 97.78
# Accuracy at 16:7 97.80
# Accuracy at 16:8 97.82
# Accuracy at 16:9 97.84
# Accuracy at 16:10 97.68
# till:16, best dev acc 97.90 at 167 with test acc 97.77, time 2.134477

# Accuracy at 17:0 97.84
# Accuracy at 17:1 97.86
# Accuracy at 17:2 98.00:97.69
# Accuracy at 17:3 97.90
# Accuracy at 17:4 97.92
# Accuracy at 17:5 97.88
# Accuracy at 17:6 97.84
# Accuracy at 17:7 97.80
# Accuracy at 17:8 97.90
# Accuracy at 17:9 97.78
# Accuracy at 17:10 97.72
# till:17, best dev acc 98.00 at 189 with test acc 97.69, time 2.140694

# Accuracy at 18:0 97.72
# Accuracy at 18:1 97.88
# Accuracy at 18:2 97.86
# Accuracy at 18:3 97.74
# Accuracy at 18:4 97.78
# Accuracy at 18:5 97.80
# Accuracy at 18:6 97.86
# Accuracy at 18:7 97.76
# Accuracy at 18:8 97.78
# Accuracy at 18:9 97.98
# Accuracy at 18:10 98.00
# till:18, best dev acc 98.00 at 189 with test acc 97.69, time 2.140415

# Accuracy at 19:0 97.98
# Accuracy at 19:1 97.98
# Accuracy at 19:2 98.06:97.82
# Accuracy at 19:3 98.04
# Accuracy at 19:4 97.98
# Accuracy at 19:5 97.92
# Accuracy at 19:6 97.88
# Accuracy at 19:7 97.98
# Accuracy at 19:8 98.02
# Accuracy at 19:9 97.94
# Accuracy at 19:10 97.94
# till:19, best dev acc 98.06 at 211 with test acc 97.82, time 2.163405


# layer:3, minibatch:100, hidden:500, keep_rate:0.9, k:30
# Accuracy at 0:0 13.62:12.23
# Accuracy at 0:1 62.76:63.45
# Accuracy at 0:2 76.30:76.64
# Accuracy at 0:3 81.84:81.88
# Accuracy at 0:4 85.18:85.29
# Accuracy at 0:5 87.76:87.59
# Accuracy at 0:6 89.34:88.98
# Accuracy at 0:7 90.24:89.83
# Accuracy at 0:8 90.08
# Accuracy at 0:9 91.18:91.00
# Accuracy at 0:10 91.62:91.06
# till:0, best dev acc 91.62 at 10 with test acc 91.06, time 7.441815

# Accuracy at 1:0 91.80:91.05
# Accuracy at 1:1 92.30:91.64
# Accuracy at 1:2 92.14
# Accuracy at 1:3 92.82:92.06
# Accuracy at 1:4 92.80
# Accuracy at 1:5 93.32:92.60
# Accuracy at 1:6 93.36:92.71
# Accuracy at 1:7 93.60:92.76
# Accuracy at 1:8 93.76:93.22
# Accuracy at 1:9 93.80:93.13
# Accuracy at 1:10 94.04:93.30
# till:1, best dev acc 94.04 at 21 with test acc 93.30, time 7.147577

# Accuracy at 2:0 94.02
# Accuracy at 2:1 94.40:93.79
# Accuracy at 2:2 94.24
# Accuracy at 2:3 94.58:93.98
# Accuracy at 2:4 94.56
# Accuracy at 2:5 94.72:94.25
# Accuracy at 2:6 94.50
# Accuracy at 2:7 94.98:94.39
# Accuracy at 2:8 95.20:94.61
# Accuracy at 2:9 95.22:94.66
# Accuracy at 2:10 95.12
# till:2, best dev acc 95.22 at 31 with test acc 94.66, time 7.136296

# Accuracy at 3:0 94.98
# Accuracy at 3:1 95.18
# Accuracy at 3:2 95.38:95.10
# Accuracy at 3:3 95.48:95.01
# Accuracy at 3:4 95.60:95.18
# Accuracy at 3:5 95.60
# Accuracy at 3:6 95.64:95.23
# Accuracy at 3:7 95.44
# Accuracy at 3:8 95.64
# Accuracy at 3:9 95.68:95.48
# Accuracy at 3:10 96.16:95.70
# till:3, best dev acc 96.16 at 43 with test acc 95.70, time 7.137904

# Accuracy at 4:0 95.92
# Accuracy at 4:1 96.06
# Accuracy at 4:2 96.06
# Accuracy at 4:3 95.94
# Accuracy at 4:4 95.90
# Accuracy at 4:5 96.10
# Accuracy at 4:6 96.12
# Accuracy at 4:7 96.12
# Accuracy at 4:8 96.26:96.01
# Accuracy at 4:9 96.52:96.06
# Accuracy at 4:10 96.38
# till:4, best dev acc 96.52 at 53 with test acc 96.06, time 7.145910

# Accuracy at 5:0 96.44
# Accuracy at 5:1 96.50
# Accuracy at 5:2 96.48
# Accuracy at 5:3 96.56:96.20
# Accuracy at 5:4 96.46
# Accuracy at 5:5 96.54
# Accuracy at 5:6 96.54
# Accuracy at 5:7 96.50
# Accuracy at 5:8 96.52
# Accuracy at 5:9 96.66:96.31
# Accuracy at 5:10 96.82:96.30
# till:5, best dev acc 96.82 at 65 with test acc 96.30, time 7.106077

# Accuracy at 6:0 96.78
# Accuracy at 6:1 96.78
# Accuracy at 6:2 96.70
# Accuracy at 6:3 96.78
# Accuracy at 6:4 96.82
# Accuracy at 6:5 96.72
# Accuracy at 6:6 96.82
# Accuracy at 6:7 96.86:96.62
# Accuracy at 6:8 96.82
# Accuracy at 6:9 96.80
# Accuracy at 6:10 96.82
# till:6, best dev acc 96.86 at 73 with test acc 96.62, time 7.093850

# Accuracy at 7:0 96.92:96.54
# Accuracy at 7:1 97.00:96.70
# Accuracy at 7:2 97.00
# Accuracy at 7:3 96.84
# Accuracy at 7:4 96.84
# Accuracy at 7:5 97.00
# Accuracy at 7:6 96.98
# Accuracy at 7:7 96.86
# Accuracy at 7:8 96.82
# Accuracy at 7:9 96.96
# Accuracy at 7:10 97.00
# till:7, best dev acc 97.00 at 78 with test acc 96.70, time 7.088891

# Accuracy at 8:0 97.00
# Accuracy at 8:1 97.06:97.01
# Accuracy at 8:2 97.04
# Accuracy at 8:3 97.14:96.82
# Accuracy at 8:4 97.14
# Accuracy at 8:5 96.98
# Accuracy at 8:6 97.06
# Accuracy at 8:7 97.20:96.88
# Accuracy at 8:8 97.28:96.93
# Accuracy at 8:9 97.10
# Accuracy at 8:10 97.26
# till:8, best dev acc 97.28 at 96 with test acc 96.93, time 7.085866

# Accuracy at 9:0 97.26
# Accuracy at 9:1 97.34:97.21
# Accuracy at 9:2 97.28
# Accuracy at 9:3 97.22
# Accuracy at 9:4 97.18
# Accuracy at 9:5 97.44:97.00
# Accuracy at 9:6 97.42
# Accuracy at 9:7 97.32
# Accuracy at 9:8 97.14
# Accuracy at 9:9 97.22
# Accuracy at 9:10 97.30
# till:9, best dev acc 97.44 at 104 with test acc 97.00, time 7.144604

# Accuracy at 10:0 97.28
# Accuracy at 10:1 97.40
# Accuracy at 10:2 97.40
# Accuracy at 10:3 97.12
# Accuracy at 10:4 97.40
# Accuracy at 10:5 97.34
# Accuracy at 10:6 97.38
# Accuracy at 10:7 97.54:97.26
# Accuracy at 10:8 97.60:97.36
# Accuracy at 10:9 97.46
# Accuracy at 10:10 97.54
# till:10, best dev acc 97.60 at 118 with test acc 97.36, time 7.166307

# Accuracy at 11:0 97.50
# Accuracy at 11:1 97.48
# Accuracy at 11:2 97.38
# Accuracy at 11:3 97.52
# Accuracy at 11:4 97.32
# Accuracy at 11:5 97.54
# Accuracy at 11:6 97.54
# Accuracy at 11:7 97.62:97.34
# Accuracy at 11:8 97.38
# Accuracy at 11:9 97.48
# Accuracy at 11:10 97.34
# till:11, best dev acc 97.62 at 128 with test acc 97.34, time 7.279715

# Accuracy at 12:0 97.72:97.46
# Accuracy at 12:1 97.72
# Accuracy at 12:2 97.56
# Accuracy at 12:3 97.50
# Accuracy at 12:4 97.66
# Accuracy at 12:5 97.56
# Accuracy at 12:6 97.52
# Accuracy at 12:7 97.52
# Accuracy at 12:8 97.50
# Accuracy at 12:9 97.64
# Accuracy at 12:10 97.50
# till:12, best dev acc 97.72 at 132 with test acc 97.46, time 7.294862

# Accuracy at 13:0 97.60
# Accuracy at 13:1 97.52
# Accuracy at 13:2 97.62
# Accuracy at 13:3 97.60
# Accuracy at 13:4 97.64
# Accuracy at 13:5 97.72
# Accuracy at 13:6 97.44
# Accuracy at 13:7 97.44
# Accuracy at 13:8 97.58
# Accuracy at 13:9 97.54
# Accuracy at 13:10 97.72
# till:13, best dev acc 97.72 at 132 with test acc 97.46, time 7.307636

# Accuracy at 14:0 97.68
# Accuracy at 14:1 97.76:97.62
# Accuracy at 14:2 97.74
# Accuracy at 14:3 97.74
# Accuracy at 14:4 97.66
# Accuracy at 14:5 97.56
# Accuracy at 14:6 97.80:97.61
# Accuracy at 14:7 97.72
# Accuracy at 14:8 97.66
# Accuracy at 14:9 97.78
# Accuracy at 14:10 97.82:97.46
# till:14, best dev acc 97.82 at 164 with test acc 97.46, time 7.321152

# Accuracy at 15:0 97.90:97.68
# Accuracy at 15:1 97.82
# Accuracy at 15:2 97.76
# Accuracy at 15:3 97.78
# Accuracy at 15:4 97.84
# Accuracy at 15:5 97.78
# Accuracy at 15:6 97.78
# Accuracy at 15:7 97.80
# Accuracy at 15:8 97.92:97.69
# Accuracy at 15:9 97.86
# Accuracy at 15:10 97.68
# till:15, best dev acc 97.92 at 173 with test acc 97.69, time 7.299692

# Accuracy at 16:0 97.80
# Accuracy at 16:1 97.86
# Accuracy at 16:2 97.74
# Accuracy at 16:3 98.04:97.67
# Accuracy at 16:4 97.88
# Accuracy at 16:5 98.02
# Accuracy at 16:6 98.02
# Accuracy at 16:7 97.94
# Accuracy at 16:8 97.72
# Accuracy at 16:9 97.86
# Accuracy at 16:10 97.80
# till:16, best dev acc 98.04 at 179 with test acc 97.67, time 7.298542

# Accuracy at 17:0 97.80
# Accuracy at 17:1 97.92
# Accuracy at 17:2 97.78
# Accuracy at 17:3 98.00
# Accuracy at 17:4 97.90
# Accuracy at 17:5 97.98
# Accuracy at 17:6 97.76
# Accuracy at 17:7 97.92
# Accuracy at 17:8 97.80
# Accuracy at 17:9 97.76
# Accuracy at 17:10 97.76
# till:17, best dev acc 98.04 at 179 with test acc 97.67, time 7.282066

# Accuracy at 18:0 97.82
# Accuracy at 18:1 97.90
# Accuracy at 18:2 97.96
# Accuracy at 18:3 97.90
# Accuracy at 18:4 97.80
# Accuracy at 18:5 97.96
# Accuracy at 18:6 97.98
# Accuracy at 18:7 98.04
# Accuracy at 18:8 98.04
# Accuracy at 18:9 97.90
# Accuracy at 18:10 97.76
# till:18, best dev acc 98.04 at 179 with test acc 97.67, time 7.256023

# Accuracy at 19:0 97.82
# Accuracy at 19:1 97.86
# Accuracy at 19:2 97.90
# Accuracy at 19:3 97.94
# Accuracy at 19:4 98.00
# Accuracy at 19:5 97.94
# Accuracy at 19:6 97.92
# Accuracy at 19:7 97.90
# Accuracy at 19:8 98.08:97.82
# Accuracy at 19:9 97.80
# Accuracy at 19:10 97.86
# till:19, best dev acc 98.08 at 217 with test acc 97.82, time 7.234164



