import tensorflow as tf
slim = tf.contrib.slim

import gin.tf

@gin.configurable
class base_layer(object):

    def __init__(self,
                 conv_outputs=[32, 64, 64],
                 conv_shapes=[[8, 8], [4, 4], [3, 3]],
                 conv_strides=[4, 2, 1],
                 fc_outputs=[512],
                 weights_initializer=tf.contrib.layers.xavier_initializer()
                 ):
        assert len(conv_outputs) == len(conv_shapes) == len(conv_strides), \
            "The lengths of conv_outputs, conv_shapes, and conv_strides are not equal."
        assert len(fc_outputs) > 0 and len(conv_outputs) > 0, \
            "The length of fc_outputs, conv_outputs must be greater than zero."
        self.agent = None
        self.conv_stack_args = [(conv_outputs[i], conv_shapes[i], conv_strides[i]) for i in range(len(conv_outputs))]
        self.fc_outputs = fc_outputs
        self.weights_initializer = weights_initializer
        self.use_lstm_net = False

    def set_agent(self, agent, batch_size):
        self.agent = agent

    def set_weights_initializer(self, weights_initializer):
        self.weights_initializer = weights_initializer

    def get_conv_layer(self, input):
        # Param : 84 x 84 x 4 x 32 x 8 x 8 =  57.8M
        # Ops : ((84 - 8 + 1) / 4 + 1)^2 * 4 * 8 * 8 * 32 = 2.9M
        with slim.arg_scope([slim.conv2d],
                            padding='SAME', weights_initializer=self.weights_initializer):
            conv2 = slim.stack(input, slim.conv2d, self.conv_stack_args, scope='conv_layer')
        return conv2

    def get_full_layer(self, input, actions, batch_size=1):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=self.weights_initializer):
            fc = slim.stack(input, slim.fully_connected, self.fc_outputs, scope='fc_layer')
            fc = slim.fully_connected(fc, int(actions), activation_fn=None)
        return fc

    def reset_state(self):
        pass
