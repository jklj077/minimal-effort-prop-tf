#pylint: disable=not-context-manager

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from meprop import MeProp
from meprop import MePropRecord

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('fake_data', False,
                        'If true, uses fake data for unit testing.')
tf.flags.DEFINE_integer('max_steps', 5500, 'Number of steps to run trainer.')
tf.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.flags.DEFINE_boolean('meprop', False, 'If true, use meprop when training')
tf.flags.DEFINE_integer('meprop_k', 160, 'k in meprop')
tf.flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
tf.flags.DEFINE_float('prune', 0.08, 'prune rate for prune')
tf.flags.DEFINE_string('data_dir', 'input_data',
                       'Directory for storing input data')
tf.flags.DEFINE_string('log_dir', 'logs/prune', 'Summaries log directory')

PRUNE_OP = 'prune_operations'
PRUNE_MASK_VAR = 'prune_mask_variables'
PRUNE_RECORD_VAR = 'prune_record_variables'
PRUNE_REF_VAR = 'prune_ref_variables'
MEPROP_TEN = 'meprop_variables'
RECORD_OP = 'record_operations'
TRAIN_SUM_OP = 'train_summary_operations'
TEST_SUM_OP = 'test_summary_operations'
PRUNE_SUM_OP = 'prune_summary_operations'


class ModelPrunable(object):
    def __init__(self, input_dim, output_dim, hidden_dims):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.train_sum_collections = []
        self.test_sum_collections = []

        self.x = tf.placeholder(tf.float32, [None, input_dim], name='x-input')
        self.y_ = tf.placeholder(
            tf.float32, [None, output_dim], name='y-input')

        self.keep_prob = tf.placeholder(tf.float32)
        self.top_k = tf.placeholder(tf.int32)
        self.train_sum(tf.summary.scalar('top_k', self.top_k))
        self.prune = tf.placeholder(tf.float32)
        self.train_sum(tf.summary.scalar('prune_rate', self.prune))
        self.train_sum(
            tf.summary.scalar('dropout_keep_probability', self.keep_prob))

        self.layers = []
        if not hidden_dims:
            hidden = self.nn_layer(
                self.x, input_dim, output_dim, 'layer', act=tf.identity)
            self.layers.append(hidden)
        else:
            for i, hidden_dim in enumerate(hidden_dims):
                if i == 0:
                    hidden = self.nn_layer_masked(
                        self.x, input_dim, hidden_dim, 'layer' + str(i + 1))
                else:
                    hidden = self.nn_layer_masked(
                        self.layers[-1], hidden_dims[i - 1], hidden_dim,
                        'layer' + str(i + 1))
                # with tf.name_scope('layer'+str(i+1)+'dropout'):
                #     dropped = tf.nn.dropout(hidden, self.keep_prob)
                #self.layers.append(dropped)
                self.layers.append(hidden)

            hidden = self.nn_layer(
                self.layers[-1],
                hidden_dims[-1],
                output_dim,
                'layer' + str(len(hidden_dims) + 1),
                act=tf.identity)
            self.layers.append(hidden)

        self.y = self.layers[-1]

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y)
            cross_entropy = tf.reduce_mean(diff)
        self.all_sum(tf.summary.scalar('cross_entropy', cross_entropy))

        with tf.name_scope('gradients'):
            trainable = tf.trainable_variables()
            meprop_ten = tf.get_collection(MEPROP_TEN)
            records = tf.get_collection(PRUNE_RECORD_VAR)
            refs = tf.get_collection(PRUNE_REF_VAR)

            all_tensor = trainable + meprop_ten + records + refs
            grads = tf.gradients(cross_entropy, all_tensor)
            grads = list(zip(grads, all_tensor))

            trainable = grads[:len(trainable)]
            grads = grads[len(trainable):]
            meprop_ten = grads[:len(meprop_ten)]
            grads = grads[len(meprop_ten):]
            records = grads[:len(records)]
            refs = grads[len(records):]

            for i, (grad, _) in enumerate(meprop_ten):
                self.train_sum(
                    tf.summary.histogram('layer' + str(i + 1) +
                                         '/matmul/gradient', grad))

        with tf.name_scope('record_stat'):
            for i, ((record_grad, record),
                    (ref_grad, ref)) in enumerate(zip(records, refs)):
                with tf.name_scope('layer' + str(i + 1)):
                    incre_record = tf.assign_add(record, record_grad)
                    tf.add_to_collection(RECORD_OP, incre_record)
                    incre_ref = tf.assign_add(ref, ref_grad)
                    tf.add_to_collection(RECORD_OP, incre_ref)

        with tf.name_scope('update_param'):
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.train_step = opt.apply_gradients(grads_and_vars=trainable)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            self.test_sum(tf.summary.scalar('accuracy', self.accuracy))

        with tf.name_scope('prune_step'):
            for i, (mask, record, ref) in enumerate(
                    zip(
                        tf.get_collection(PRUNE_MASK_VAR),
                        tf.get_collection(PRUNE_RECORD_VAR),
                        tf.get_collection(PRUNE_REF_VAR))):
                with tf.name_scope('layer' + str(i)):
                    self.prune_update(mask, record, ref)

        # Merge all the summaries
        self.prune_merged = tf.summary.merge(tf.get_collection(PRUNE_SUM_OP))
        self.train_merged = tf.summary.merge(tf.get_collection(TRAIN_SUM_OP))
        self.test_merged = tf.summary.merge(tf.get_collection(TEST_SUM_OP))
        self.record_stat = tf.get_collection(RECORD_OP)
        self.prune_op = tf.get_collection(PRUNE_OP)

    def train_sum(self, tensor):
        tf.add_to_collection(TRAIN_SUM_OP, tensor)

    def test_sum(self, tensor):
        tf.add_to_collection(TEST_SUM_OP, tensor)

    def all_sum(self, tensor):
        tf.get_default_graph().add_to_collections([TRAIN_SUM_OP, TEST_SUM_OP],
                                                  tensor)

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            self.train_sum(tf.summary.scalar('mean', mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            self.train_sum(tf.summary.scalar('stddev', stddev))
            self.train_sum(tf.summary.scalar('max', tf.reduce_max(var)))
            self.train_sum(tf.summary.scalar('min', tf.reduce_min(var)))
            self.train_sum(tf.summary.histogram('dist', var))

    def prune_update(self, mask, record, ref):
        with tf.name_scope('update_mask'):
            min_val = tf.cast(self.prune * tf.cast(ref, tf.float32), tf.int32)
            m = tf.greater(record, min_val)
            update_mask = tf.assign(mask, tf.cast(m, tf.float32))
        with tf.control_dependencies([update_mask]):
            with tf.name_scope('reset_stat'):
                clear_record = tf.assign(record, record * 0)
                clear_ref = tf.assign(ref, ref * 0)
        tf.add_to_collection(PRUNE_OP, update_mask)
        tf.add_to_collection(PRUNE_OP, clear_record)
        tf.add_to_collection(PRUNE_OP, clear_ref)
        tf.add_to_collection(PRUNE_SUM_OP,
                             tf.summary.scalar('min_hit', min_val))
        tf.add_to_collection(PRUNE_SUM_OP,
                             tf.summary.histogram('updated_mask', update_mask))

    def nn_layer(self,
                 input_tensor,
                 input_dim,
                 output_dim,
                 layer_name,
                 act=tf.nn.relu,
                 meprop=False):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                matmul = tf.matmul(input_tensor, weights)
                tf.add_to_collection(MEPROP_TEN, matmul)
                if meprop:
                    prebias = MeProp(matmul, self.top_k)
                else:
                    prebias = matmul
                preactivate = prebias + biases
                self.all_sum(
                    tf.summary.histogram('pre_activations', preactivate))
            activations = act(preactivate, name='activation')
            self.all_sum(tf.summary.histogram('activations', activations))
        return activations

    def nn_layer_masked(self,
                        input_tensor,
                        input_dim,
                        output_dim,
                        layer_name,
                        act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.name_scope('masks'):
                mask = tf.Variable(tf.ones([output_dim]), trainable=False)
                length = tf.count_nonzero(mask)
                self.all_sum(tf.summary.histogram('mask', mask))
                self.all_sum(tf.summary.scalar('length', length))
            with tf.name_scope('backprop_top_record'):
                record = tf.Variable(
                    tf.zeros([output_dim], dtype=tf.int32), trainable=False)
                self.train_sum(
                    tf.summary.histogram('record_histogram', record))
                ref = tf.Variable(
                    tf.constant(0, dtype=tf.int32), trainable=False)
                self.train_sum(tf.summary.scalar('record_times', ref))
            tf.add_to_collection(PRUNE_MASK_VAR, mask)
            tf.add_to_collection(PRUNE_RECORD_VAR, record)
            tf.add_to_collection(PRUNE_REF_VAR, ref)
            with tf.name_scope('Wx_plus_b'):
                matmul = tf.matmul(input_tensor, weights)
                tf.add_to_collection(MEPROP_TEN, matmul)
                prebias = MePropRecord(matmul, self.top_k, record, ref)
                preactivate = prebias + biases
                self.all_sum(
                    tf.summary.histogram('pre_activations', preactivate))
            with tf.name_scope('activation'):
                activations = act(preactivate)
                self.all_sum(tf.summary.histogram('activations', activations))
                masked = mask * activations
                self.all_sum(
                    tf.summary.histogram('masked_activations', masked))
        return masked


def train():
    # Import data
    mnist = input_data.read_data_sets(
        FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()

    model = ModelPrunable(784, 10, [500, 500])
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    dev_writer = tf.summary.FileWriter(FLAGS.log_dir + '/dev')
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train, dev):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
            top_k = FLAGS.meprop_k
            prune = FLAGS.prune
        elif dev:
            xs, ys = mnist.validation.images, mnist.validation.labels
            top_k = FLAGS.meprop_k
            k = 1.0
            prune = FLAGS.prune
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
            top_k = FLAGS.meprop_k
            prune = FLAGS.prune
        return {
            model.x: xs,
            model.y_: ys,
            model.keep_prob: k,
            model.top_k: top_k,
            model.prune: prune
        }

    # one step is 100 examples, one epoch is 55000/100 = 550 steps
    for i in range(FLAGS.max_steps):
        if (i + 1) % 100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _, _ = sess.run(
                [model.train_merged, model.train_step, model.record_stat],
                feed_dict=feed_dict(True, False),
                options=run_options,
                run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step{}'.format(i + 1))
            train_writer.add_summary(summary, i*2)
            print('Step {}: add run metadata'.format(i))
        else:
            summary, _, _ = sess.run(
                [model.train_merged, model.train_step, model.record_stat],
                feed_dict=feed_dict(True, False))
            train_writer.add_summary(summary, i)
        if (i + 1) % 10 == 0:
            summary, acc = sess.run(
                [model.test_merged, model.accuracy],
                feed_dict=feed_dict(False, True))
            dev_writer.add_summary(summary, i * 2)
            print('Step {}: dev acc: {:.2f}'.format(i, acc*100), end='')
            summary, acc = sess.run(
                [model.test_merged, model.accuracy],
                feed_dict=feed_dict(False, False))
            test_writer.add_summary(summary, i * 2)
            print(', test acc: {:.2f}'.format(acc*100))
        if (i + 1) % 50 == 0:
            summary, _ = sess.run(
                [model.prune_merged, model.prune_op],
                feed_dict={model.prune: FLAGS.prune})
            train_writer.add_summary(summary, i * 2 + 1)
            print('Step {}: prune'.format(i))
            summary, acc = sess.run(
                [model.test_merged, model.accuracy],
                feed_dict=feed_dict(False, True))
            dev_writer.add_summary(summary, i * 2 + 1)
            print('Step {}: dev acc: {:.2f}'.format(i, acc*100), end='')
            summary, acc = sess.run(
                [model.test_merged, model.accuracy],
                feed_dict=feed_dict(False, False))
            test_writer.add_summary(summary, i * 2 + 1)
            print(', test acc: {:.2f}'.format(acc*100))
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
