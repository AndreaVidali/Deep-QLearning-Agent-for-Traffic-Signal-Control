import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # now setup the model
        self._define_model()

    # DEFINE THE STRUCTURE OF THE NEURAL NETWORK
    def _define_model(self):
        # placeholders
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

        # list of nn layers
        fc1 = tf.layers.dense(self._states, 400, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 400, activation=tf.nn.relu)
        fc4 = tf.layers.dense(fc3, 400, activation=tf.nn.relu)
        fc5 = tf.layers.dense(fc4, 400, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc5, self._num_actions)

        # parameters
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A SINGLE STATE
    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A BATCH OF STATES
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    # TRAIN THE NETWORK
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
