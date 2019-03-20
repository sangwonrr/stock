
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from train_data import  trade_train_data
import numpy as np
import gin.tf


@gin.configurable
class stockPreprocessing(object):

  def __init__(self, codes, skip=10, data_size=7):
    self.trade_train_data = trade_train_data(codes[0], 'train', skip)
    self.skip = skip
    self.data_size = data_size
    self.purchase_price = 0.0
    self.standard_price = 0.0
    self.last_observation = None

  @property
  def observation_space(self):
    return (10,8)

  @property
  def action_space(self):
    return (21,)

  def reset(self):
    self.purchase_price = 0.0
    self.standard_price = 0.0
    self.last_observation = None
    return np.zeros(self.observation_space, dtype=np.float32)

  def step(self, actions):
    reward = 0.

    # 0~9 buy, 10 ~ 19 sell, 20 maintain
    action_num = np.argmax(actions)
    if action_num != 20 and self.last_observation is not None:
      select_price = self.last_observation[action_num % 10][5]
      if action_num >= 10: #sell
        if self.purchase_price > 0:
          reward += max(0.0, select_price - self.purchase_price)
      else:
        if self.purchase_price == 0:
          self.purchase_price = select_price
        else :
          self.purchase_price = (self.purchase_price + select_price) / 2

    done, observation = self.trade_train_data.get_input_one_data()
    if done:
      observation = np.zeros(self.observation_space, dtype=np.float32)
      select_price = self.last_observation[-1][5]
      if self.purchase_price > 0:
        reward += max(0.0, select_price - self.purchase_price)
      if self.trade_train_data.next_data() == False:
        self.trade_train_data.shuffle_min_data()
    else:
      purchase_price_arr = np.full((10, 1), self.purchase_price)
      observation = np.concatenate([observation, purchase_price_arr], axis=-1)

    self.last_observation = observation
    return observation, reward, done