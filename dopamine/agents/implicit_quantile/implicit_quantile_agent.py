# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import math


from dopamine.agents.rainbow import rainbow_agent
import numpy as np
import tensorflow as tf
from dopamine.agents import base_layer

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class ImplicitQuantileAgent(rainbow_agent.RainbowAgent):
  """An extension of Rainbow to perform implicit quantile regression."""

  def __init__(self,
               sess,
               num_actions,
               kappa=1.0,
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               double_dqn=False,
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int[], number of actions the agent can take at any state.
      kappa: float, Huber loss cutoff.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.kappa = kappa
    # num_tau_samples = N below equation (3) in the paper.
    self.num_tau_samples = num_tau_samples
    # num_tau_prime_samples = N' below equation (3) in the paper.
    self.num_tau_prime_samples = num_tau_prime_samples
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.double_dqn = double_dqn

    super(ImplicitQuantileAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

    self.neural_layer.net_output_batch_size = num_quantile_samples

  def _get_network_type(self):
    """Returns the type of the outputs of the implicit quantile network.

    Returns:
      _network_type object defining the outputs of the network.
    """
    return collections.namedtuple(
        'iqn_network', ['quantile_values', 'quantiles'])

  def _network_template(self, state, num_quantiles, batch_size):
    r"""Builds an Implicit Quantile ConvNet.

    Takes state and quantile as inputs and outputs state-action quantile values.

    Args:
      state: A `tf.placeholder` for the RL state.
      num_quantiles: int, number of quantile inputs.

    Returns:
      _network_type object containing quantile value outputs of the network.
    """

    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    state_net = tf.cast(state, tf.float32)
    # state_net = tf.div(state_net, 255.)
    state_net = slim.flatten(state_net)
    state_net_size = state_net.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

    batch_size = state_net.get_shape().as_list()[0]
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random_uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                        weights_initializer=weights_initializer)
    # Hadamard product.
    net = tf.multiply(state_net_tiled, quantile_net)
    net = self.neural_layer.get_full_layer(net, np.sum(self.num_actions, dtype=int),
                                           num_quantiles * batch_size, num_quantiles)
    quantile_values = []
    startindex = 0
    for i in range(len(self.num_actions)):
      quantile_values.append(tf.slice(net, [0, startindex], [-1, self.num_actions[i]]))
      startindex = self.num_actions[i]

    return self._get_network_type()(quantile_values=quantile_values,
                                    quantiles=quantiles)

  def _build_networks(self):
    """Builds the IQN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,
                                            self.num_quantile_samples,
                                            1)
    # Shape of self._net_outputs.quantile_values:
    # num_quantile_samples x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    self._q_values = [tf.reduce_mean(self._net_outputs.quantile_values[i], axis=0)
                      for i in range(len(self.num_actions))]
    self._q_argmax = [tf.argmax(self._q_values[i], axis=0) for i in range(len(self.num_actions))]

    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   self.num_tau_samples,
                                                   self._replay.batch_size)
    # Shape: (num_tau_samples x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    self._replay_net_quantiles = self._replay_net_outputs.quantiles

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.target_convnet(
        self._replay.next_states, self.num_tau_prime_samples, self._replay.batch_size)
    # Shape: (num_tau_prime_samples x batch_size) x num_actions.
    vals = self._replay_net_target_outputs.quantile_values
    self._replay_net_target_quantile_values = vals

    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    if self.double_dqn:
      outputs_action = self.online_convnet(self._replay.next_states,
                                           self.num_quantile_samples,
                                           self._replay.batch_size)
    else:
      outputs_action = self.target_convnet(self._replay.next_states,
                                           self.num_quantile_samples,
                                           self._replay.batch_size)

    # Shape: (num_quantile_samples x batch_size) x num_actions.
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = [tf.reshape(outputs_action.quantile_values[i],
                                               [self.num_quantile_samples,
                                                self._replay.batch_size,
                                                self.num_actions[i]])
                                     for i in range(len(self.num_actions))]
    # Shape: batch_size x num_actions.
    self._replay_net_target_q_values = [tf.squeeze(tf.reduce_mean(
        target_quantile_values_action[i], axis=0)) for i in range(len(self.num_actions))]
    self._replay_next_qt_argmax = [tf.argmax(self._replay_net_target_q_values[i], axis=1)
                                   for i in range(len(self.num_actions))]

  def _build_target_quantile_values_op(self, index):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
    rewards = self._replay.rewards[:, None]
    rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

    is_terminal_multiplier = 1. - tf.to_float(self._replay.terminals)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_tau_prime_samples, 1])

    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.

    replay_next_qt_argmax = tf.tile(
        self._replay_next_qt_argmax[index][:, None], [self.num_tau_prime_samples, 1])

    # Shape of batch_indices: (num_tau_prime_samples x batch_size) x 1.
    batch_indices = tf.cast(tf.range(
        self.num_tau_prime_samples * batch_size)[:, None], tf.int64)

    # Shape of batch_indexed_target_values:
    # (num_tau_prime_samples x batch_size) x 2.
    batch_indexed_target_values = tf.concat(
        [batch_indices, replay_next_qt_argmax], axis=1)

    # Shape of next_target_values: (num_tau_prime_samples x batch_size) x 1.
    target_quantile_values = tf.gather_nd(
        self._replay_net_target_quantile_values[index],
        batch_indexed_target_values)[:, None]

    return rewards + gamma_with_terminal * target_quantile_values

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    loss_avg = 0
    for i in range(len(self.num_actions)):
        target_quantile_values = tf.stop_gradient(
            self._build_target_quantile_values_op(i))
        # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
        # the manner in which the target_quantile_values are tiled.
        target_quantile_values = tf.reshape(target_quantile_values,
                                            [self.num_tau_prime_samples,
                                             batch_size, 1])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_prime_samples x 1 to prepare for computation of
        # Bellman errors.
        # Final shape of target_quantile_values:
        # batch_size x num_tau_prime_samples x 1.
        target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])

        # Shape of indices: (num_tau_samples x batch_size) x 1.
        # Expand dimension by one so that it can be used to index into all the
        # quantiles when using the tf.gather_nd function (see below).
        indices = tf.range(self.num_tau_samples * batch_size)[:, None]

        # Expand the dimension by one so that it can be used to index into all the
        # quantiles when using the tf.gather_nd function (see below).
        reshaped_actions = self._replay.actions[:, None, i]
        reshaped_actions = tf.tile(reshaped_actions, [self.num_tau_samples, 1])
        # Shape of reshaped_actions: (num_tau_samples x batch_size) x 2.
        reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

        chosen_action_quantile_values = tf.gather_nd(
            self._replay_net_quantile_values[i], reshaped_actions)
        # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
        # in which the quantile values are tiled.
        chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                                   [self.num_tau_samples,
                                                    batch_size, 1])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_samples x 1 to prepare for computation of
        # Bellman errors.
        # Final shape of chosen_action_quantile_values:
        # batch_size x num_tau_samples x 1.
        chosen_action_quantile_values = tf.transpose(
            chosen_action_quantile_values, [1, 0, 2])

        # Shape of bellman_erors and huber_loss:
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        bellman_errors = target_quantile_values[
            :, :, None, :] - chosen_action_quantile_values[:, None, :, :]
        # The huber loss (see Section 2.3 of the paper) is defined via two cases:
        # case_one: |bellman_errors| <= kappa
        # case_two: |bellman_errors| > kappa
        huber_loss_case_one = tf.to_float(
            tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
        huber_loss_case_two = tf.to_float(
            tf.abs(bellman_errors) > self.kappa) * self.kappa * (
                tf.abs(bellman_errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two

        # Reshape replay_quantiles to batch_size x num_tau_samples x 1
        replay_quantiles = tf.reshape(
            self._replay_net_quantiles, [self.num_tau_samples, batch_size, 1])
        replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])

        # Tile by num_tau_prime_samples along a new dimension. Shape is now
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        # These quantiles will be used for computation of the quantile huber loss
        # below (see section 2.3 of the paper).
        replay_quantiles = tf.to_float(tf.tile(
            replay_quantiles[:, None, :, :], [1, self.num_tau_prime_samples, 1, 1]))
        # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
        quantile_huber_loss = (tf.abs(replay_quantiles - tf.stop_gradient(
            tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa
        # Sum over current quantile value (num_tau_samples) dimension,
        # average over target quantile value (num_tau_prime_samples) dimension.
        # Shape: batch_size x num_tau_prime_samples x 1.
        loss = tf.reduce_sum(quantile_huber_loss, axis=2)
        # Shape: batch_size x 1.
        loss_avg += tf.reduce_mean(loss, axis=1)
    loss_avg /= len(self.num_actions)
    # TODO(kumasaurabh): Add prioritized replay functionality here.
    if self._replay_scheme == 'prioritized':
        # The original prioritized experience replay uses a linear exponent
        # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
        # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
        # a fixed exponent actually performs better, except on Pong.
        probs = self._replay.transition['sampling_probabilities']
        loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
        loss_weights /= tf.reduce_max(loss_weights)

        # Rainbow and prioritized replay are parametrized by an exponent alpha,
        # but in both cases it is set to 0.5 - for simplicity's sake we leave it
        # as is here, using the more direct tf.sqrt(). Taking the square root
        # "makes sense", as we are dealing with a squared loss.
        # Add a small nonzero value to the loss to avoid 0 priority items. While
        # technically this may be okay, setting all items to 0 priority will cause
        # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
        update_priorities_op = self._replay.tf_set_priority(
            self._replay.indices, tf.sqrt(tf.squeeze(loss_avg) + 1e-10))

        # Weight the loss by the inverse priorities.
        loss_avg = loss_weights * loss_avg
    else:
        update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss_avg))
      return self.optimizer.minimize(tf.reduce_mean(loss_avg)), tf.reduce_mean(loss_avg)
