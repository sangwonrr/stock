import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from dopamine.agents import base_layer
slim = tf.contrib.slim

import gin.tf

@gin.configurable
class extend_layer(base_layer.base_layer):

    def __init__(self,
                 conv_outputs=[32, 64, 64],
                 conv_shapes=[[8, 8], [4, 4], [3, 3]],
                 conv_strides=[4, 2, 1],
                 fc_outputs=[512],
                 use_noisy_net=False,
                 use_inception_net=False,
                 use_lstm_net=False,
                 use_impala=False,
                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                 ):
        super(extend_layer, self).__init__(
            conv_outputs=conv_outputs,
            conv_shapes=conv_shapes,
            conv_strides=conv_strides,
            fc_outputs=fc_outputs,
            weights_initializer=weights_initializer)

        self.net_output_batch_size = 1
        self.use_noisy_net = use_noisy_net
        self.use_inception_net = use_inception_net
        self.use_lstm_net = use_lstm_net
        self.use_impala = use_impala
        if self.use_lstm_net:
            self.rnn_state_op = None
            self.rnn_state = None
            self.train_rnn_state = None
            self.rnn_final_state = []


    def set_agent(self, agent, batch_size):
        super().set_agent(agent)
        if self.use_lstm_net:
            self.reset_state()
            self.train_rnn_state = []
            split = len(agent.num_actions)
            for _ in range(split):
                self.train_rnn_state.append(
                    [(np.zeros([batch_size, size//split]), np.zeros([batch_size, size//split])) for size in self.fc_outputs])
            self.train_rnn_state = tuple(self.train_rnn_state)

    def reset_state(self):
        split = len(self.agent.num_actions)
        self.rnn_state = []
        for _ in range(split):
            self.rnn_state.append([(np.zeros([self.net_output_batch_size, size//split]), np.zeros([self.net_output_batch_size, size//split]))
                              for size in self.fc_outputs])

    def get_conv_layer(self, input):
        if self.use_inception_net:
            return self.Inception(input)
        elif self.use_impala:
            return self.impala_cnn(input, self.conv_stack_args)
        else:
            return super().get_conv_layer(input)

    def get_full_layer(self, input, actions, batch_size, net_output_batch_size=1):
        if self.use_noisy_net:
            with tf.variable_scope('noisy_net'):
                noisy_net = input
                count = 0
                for output_size in self.fc_outputs:
                    noisy_net = self.noisy_dense(noisy_net, output_size, name="noisy_"+str(count))
                    count += 1
                noisy_net = self.noisy_dense(
                    noisy_net,
                    actions,
                    name="noisy_output",
                    activation_fn=None)

            return noisy_net
        elif self.use_lstm_net:
            return self.output_lstm(input, actions, batch_size, net_output_batch_size)
        else:
            return super().get_full_layer(input, actions)

    def output_lstm(self, input, actions, batch_size, net_output_batch_size):
        convFlat = tf.reshape(input, [batch_size, self.agent.stack_size, -1])
        split = len(self.agent.num_actions)
        outputs_data = []
        if batch_size == net_output_batch_size:
            self.rnn_state_op = [None] * split
            self.rnn_final_state = [None] * split

        for idx in range(split):
            with tf.variable_scope(f"action_split_{idx}"):
                cells = [tf.nn.rnn_cell.LSTMCell(size / split, initializer=self.weights_initializer) for size in self.fc_outputs]
                cells = rnn.MultiRNNCell(cells)
                if batch_size == net_output_batch_size:
                    self.rnn_final_state[idx] = cells.zero_state(batch_size, tf.float32)
                    outputs, self.rnn_state_op[idx] = tf.nn.dynamic_rnn(cells, convFlat,
                                                                   initial_state=self.rnn_final_state[idx], dtype=tf.float32)
                else:
                    outputs, _ = tf.nn.dynamic_rnn(cells, convFlat, dtype=tf.float32)
                outputs = tf.reshape(outputs, [batch_size, -1])
                temp_actions = actions // np.sum(self.agent.num_actions) * self.agent.num_actions[idx]
                outputs_data.append(slim.fully_connected(outputs, int(temp_actions), activation_fn=None))
        outputs_data = tf.concat(outputs_data, axis=-1)
        return outputs_data

    def noisy_dense(self, x, num_outputs, name='Noisy', weights_initializer=None, activation_fn=tf.nn.relu):
        with tf.variable_scope(name):
            input_shape = [x.get_shape()[-1].value, num_outputs]
            mu_w = self._mu_variable('w1', input_shape)
            sig_w = self._sigma_variable('b1', input_shape)
            mu_b = self._mu_variable('w2', [input_shape[1]])
            sig_b = self._sigma_variable('b2', [input_shape[1]])

            is_trainable = tf.get_variable("istrainable", (), dtype=tf.bool,
                                           initializer=tf.constant_initializer(self.agent.eval_mode == False, dtype=tf.bool))
            # const_init = tf.constant_initializer(self.eval_mode == False, dtype=tf.bool)
            # is_trainable = tf.get_variable("istrainable", (0,), dtype=tf.bool,
            #                                initializer=const_init)
            h_fc1 = self._noisy_dense(x, input_shape, mu_w, sig_w, mu_b, sig_b, is_trainable)
            if not activation_fn is None:
                h_fc1 = activation_fn(h_fc1)
            return h_fc1

    def _mu_variable(self, name, shape):
        value = tf.constant(3, dtype=tf.float32)
        mu_init = tf.random_uniform_initializer(minval=-tf.sqrt(value / shape[0]), maxval=tf.sqrt(value / shape[0]))
        return tf.get_variable(name + "/mu", shape, initializer=mu_init)

    def _sigma_variable(self, name, shape):
        return tf.get_variable(name + "/sigma", shape, initializer=tf.constant_initializer(0.017))

    def _noisy_dense(self, input_, input_shape, mu_w, sig_w, mu_b, sig_b, is_train_process):
        eps_w = tf.cond(is_train_process, lambda: tf.random_normal(input_shape), lambda: tf.zeros(input_shape))
        eps_b = tf.cond(is_train_process, lambda: tf.random_normal([input_shape[1]]),
                        lambda: tf.zeros([input_shape[1]]))

        w_fc = tf.add(mu_w, tf.multiply(sig_w, eps_w))
        b_fc = tf.add(mu_b, tf.multiply(sig_b, eps_b))

        return tf.matmul(input_, w_fc) + b_fc

    def Inception(self, inputs):
        with slim.arg_scope([slim.conv2d],
                            stride=1, padding='SAME', weights_initializer=self.weights_initializer):
            with tf.variable_scope('NIN_1'):
                net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                  padding='VALID', scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                                  scope='Conv2d_2a_3x3')
                net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3', )


            with tf.variable_scope('NIN_2'):
                branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                           scope='MaxPool_0a_3x3')
                branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                       scope='Conv2d_0a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])

            with tf.variable_scope('NIN_3'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                           scope='Conv2d_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])

            with tf.variable_scope('NIN_4'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])

            # for idx in range(4):
            #     block_scope = 'NIN_5' + chr(ord('b') + idx)
            #     net = self.block_inception_a(net, block_scope)
            #
            # net = self.block_reduction_a(net, 'NIN_6')

        return net

    def block_inception_a(self, inputs, scope=None, reuse=None):
        """Builds Inception-A block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def block_reduction_a(self, inputs, scope=None, reuse=None):
        """Builds Reduction-A block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    def impala_cnn(self, images, conv_stack_args, use_batch_norm=False):
        """
        Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
        Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
        """

        def conv_layer(out, depth):
            out = tf.layers.conv2d(out, depth, 3, padding='same')

            if use_batch_norm:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

            return out

        def residual_block(inputs):
            depth = inputs.get_shape()[-1].value

            out = tf.nn.relu(inputs)

            out = conv_layer(out, depth)
            out = tf.nn.relu(out)
            out = conv_layer(out, depth)
            return out + inputs

        def conv_sequence(inputs, depth, strides):
            out = conv_layer(inputs, depth)
            out = tf.layers.max_pooling2d(out, pool_size=3, strides=strides, padding='same')
            out = residual_block(out)
            out = residual_block(out)
            return out

        out = images
        for output, shape, stride in conv_stack_args:
            out = conv_sequence(out, output, stride)

        return out