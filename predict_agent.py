import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
slim = tf.contrib.slim

class predict_agent():

    def __init__(self,
                 sess,
                 input_shape,
                 label_shape,
                 batch_size,
                 has_save_data,
                 output_dim=1,
                 layer_shapes=[512, 512, 512],
                 optimizer=tf.train.AdamOptimizer(
                     learning_rate=0.0001,
                     epsilon=0.00015),
                 ):
        self.sess = sess
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.layer_shapes = layer_shapes
        self.optimizer = optimizer
        self.weights_initializer = slim.variance_scaling_initializer(
            factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

        with tf.device('/cpu:*'):
            self._input_ph = tf.placeholder(tf.float32, input_shape, name='input_ph')
            self._label_ph = tf.placeholder(tf.float32, label_shape, name='label_ph')
            self._create_network()
            self._train_op, self.loss = self._build_train_op()

        if has_save_data:
            self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        cells = [tf.nn.rnn_cell.LSTMCell(size, initializer=self.weights_initializer) for size in self.layer_shapes]
        cells = rnn.MultiRNNCell(cells)
        self.rnn_final_state = cells.zero_state(self.batch_size, tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(cells, self._input_ph,
                                                       initial_state=self.rnn_final_state, dtype=tf.float32)
        outputs = tf.reshape(outputs, [self.batch_size, -1])
        self.q_value = slim.fully_connected(outputs, self.output_dim, activation_fn=None)

    def _build_train_op(self):
        loss = tf.losses.absolute_difference(self._label_ph, self.q_value)
        return self.optimizer.minimize(loss), loss

    def train(self, inputs, labels):
        _, loss = self.sess.run([self._train_op, self.loss], {self._input_ph: inputs, self._label_ph: labels})
        return loss

    def predict(self, inputs):
        predict_value = self.sess.run(self.q_value, {self._input_ph: inputs})
        return predict_value

    def predict(self, inputs, labels):
        predict_value, loss = self.sess.run([self.q_value, self.loss], {self._input_ph: inputs, self._label_ph: labels})
        return predict_value, loss