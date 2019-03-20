import numpy as np
import os
import re
import random
import functools

class train_data():
    def __init__(self, code, mode, input_data_length=30, eval_ratio=0.2):
        self.code = code
        self.mode = mode
        self._base_dir = os.path.dirname(os.path.realpath(__file__))
        self.min_data = None
        self.next_min_pos = 0
        self.input_data_length = input_data_length
        self.last_low_price = 0
        self.last_high_price = 0
        self.today_low_price = 0
        self.today_high_price = 0
        self.moving_average_5 = 0
        self.moving_average_20 = 0
        self.standard_day = 0
        self.standard_hour = 0
        self.hour_data = {}
        self.min_data_next_pos = 0
        self.eval_ratio = eval_ratio

        self.split_min_data()
        self.read_hour_data()
        self.read_daily_data()
        self.next_data()


    def set_mode(self, mode):
        self.reset()
        self.mode = mode

    def reset(self):
        self.next_min_pos = 0
        self.low_price = 0
        self.high_price = 0
        self.standard_day = 0
        self.standard_hour = 0
        self.min_data_next_pos = 0

    def get_daily_dir(self):
        return os.path.join(self._base_dir, f'./data/{self.code}')

    def get_data_dir(self):
        return os.path.join(self._base_dir, './data')

    def read_daily_data(self):
        read_data = np.load(os.path.join(self.get_data_dir(), f'{self.code}_day.npy'))
        self.daily_data = {}
        for idx in range(len(read_data)):
            self.daily_data[read_data[idx][0]] = read_data[idx][2:]

    def split_min_data(self):
        file_list = []
        for root, dirs, files in os.walk(f'{self.get_data_dir()}/{self.code}'):
            for fname in files:
                file_list.append(os.path.join(root, fname))

        random.shuffle(file_list)
        eval_length = int(len(file_list) * self.eval_ratio)
        self.eval_filse = file_list[:eval_length]
        self.train_files = file_list[eval_length:]

    def shuffle_min_data(self):
        self.reset()
        random.shuffle(self.train_files)
        self.next_data()

    def read_hour_data(self):
        read_data = np.load(os.path.join(self.get_data_dir(), f'{self.code}_hour.npy'))
        read_data = list(read_data)
        def hour_sort(x, y):
            if x[0] > y[0]:
                return 1
            elif x[0] < y[0]:
                return -1
            else:
                if x[1] > y[1]:
                    return 1
                else:
                    return -1
        read_data = sorted(read_data, key=functools.cmp_to_key(hour_sort))
        self.hour_data = {}
        for idx in range(len(read_data)):
            key = read_data[idx][0]
            if self.hour_data.get(key) == None:
                self.hour_data[key] = []
            self.hour_data[key].extend([read_data[idx][1:]])

    def next_data(self):
        idx = self.next_min_pos
        filePath = ''
        if self.mode == 'eval':
            if idx == len(self.eval_filse):
                return True
            self.min_data = np.load(self.eval_filse[idx])
            filePath = self.eval_filse[idx]
        else:
            if idx == len(self.train_files):
                return True
            self.min_data = np.load(self.train_files[idx])
            filePath = self.train_files[idx]
        self.next_min_pos += 1

        self.init_data_setting(filePath)
        return False

    def init_data_setting(self, filePath):
        regex = re.compile(f'{self.code}_(?P<date>\d*)_(?P<time>\d*)[.]?')
        find = regex.search(filePath)
        if find == None:
            print(f"{filePath} - regex error")
            return
        self.standard_day = float(find.group("date"))
        self.standard_hour = float(find.group("time"))
        daily_data_keys = list(self.daily_data.keys())
        daily_data_keys.sort()
        if self.standard_day in daily_data_keys:
            key_value = daily_data_keys[daily_data_keys.index(self.standard_day) - 1]
        else:
            key_value = daily_data_keys[daily_data_keys.index(self.standard_day-1) - 1]

        self.min_data_next_pos = 0
        self.norm_max = self.daily_data[key_value][1] * 2
        self.last_low_price = self.daily_data[key_value][1] / self.norm_max
        self.last_high_price = self.daily_data[key_value][0] / self.norm_max
        self.moving_average_5 = self.daily_data[key_value][3] / self.norm_max
        self.moving_average_20 = self.daily_data[key_value][4] / self.norm_max
        self.today_low_price = self.min_data[0][1] / self.norm_max
        self.today_high_price = self.min_data[0][1] / self.norm_max
        for key, hour_contexts in self.hour_data.items():
            for context in hour_contexts:
                if self.standard_hour <= context[0]:
                    break

                self.today_low_price = min(self.today_low_price,  context[1] / self.norm_max)
                self.today_high_price = max(self.today_high_price, context[2] / self.norm_max)

class predict_train_data(train_data):
    def __init__(self, code, mode, input_data_length=30):

        super(predict_train_data, self).__init__(
            code,
            mode,
            input_data_length)

    def reset(self):
        self.next_min_pos = 0
        self.low_price = 0
        self.high_price = 0
        self.standard_day = 0
        self.standard_hour = 0
        self.min_data_next_pos = 0

    def get_batch_size(self):
        return len(self.min_data) - self.input_data_length

    def get_input_shape(self):
        return (len(self.min_data) - self.input_data_length, self.input_data_length, 7)

    def get_label_shape(self):
        return (len(self.min_data) - self.input_data_length, 1)

    def get_input_batch_data(self):
        batch_data = []
        batch_label = []
        while(True):
            done, input_data, label = self.get_input_one_data()
            if done:
                break
            batch_data.append(input_data)
            batch_label.append(label)
        return tuple(batch_data), tuple(batch_label)

    def get_input_one_data(self):
        input_data = []
        label = []
        label_pos = self.min_data_next_pos + self.input_data_length
        if  label_pos >= len(self.min_data):
            return True, input_data, label

        label.append(self.min_data[label_pos][1] / self.norm_max)

        for context in self.min_data[self.min_data_next_pos: label_pos]:
            temp_data = []
            temp_data.append(self.last_high_price)
            temp_data.append(self.last_low_price)
            temp_data.append(self.moving_average_5)
            temp_data.append(self.moving_average_20)
            self.today_low_price = min(self.today_low_price, context[1] / self.norm_max)
            self.today_high_price = max(self.today_high_price, context[1] / self.norm_max)
            temp_data.append((self.today_low_price + self.today_high_price) / 2)
            temp_data.append(context[1] / self.norm_max)
            temp_data.append(context[2])

            input_data.append(tuple(temp_data))

        self.min_data_next_pos += 1
        return False, tuple(input_data), tuple(label)

class trade_train_data(train_data):

    def __init__(self, code, mode, input_data_length=10):
        self.interval = input_data_length
        super(trade_train_data, self).__init__(
            code,
            mode,
            input_data_length,
            0)

    def reset(self):
        self.next_min_pos = 0
        self.low_price = 0
        self.high_price = 0
        self.standard_day = 0
        self.standard_hour = 0
        self.min_data_next_pos = 0

    # def get_batch_size(self):
    #     return len(self.min_data) / self.input_data_length

    def get_input_shape(self):
        return (7,)

    # def get_input_batch_data(self):
    #     batch_data = []
    #     batch_label = []
    #     while(True):
    #         done, input_data, label  = self.get_input_one_data()
    #         if done:
    #             break
    #         batch_data.append(input_data)
    #         batch_label.append(label)
    #     return tuple(batch_data), tuple(batch_label)

    def get_input_one_data(self):
        input_data = []
        end_pos = self.min_data_next_pos + self.input_data_length
        if  end_pos >= len(self.min_data):
            return True, input_data

        for context in self.min_data[self.min_data_next_pos: end_pos]:
            temp_data = []
            temp_data.append(self.last_high_price)
            temp_data.append(self.last_low_price)
            temp_data.append(self.moving_average_5)
            temp_data.append(self.moving_average_20)
            self.today_low_price = min(self.today_low_price, context[1] / self.norm_max )
            self.today_high_price = max(self.today_high_price, context[1] / self.norm_max)
            temp_data.append((self.today_low_price + self.today_high_price) / 2)
            temp_data.append(context[1] / self.norm_max)
            temp_data.append(context[2])

            input_data.append(tuple(temp_data))

        self.min_data_next_pos += 1
        return False, tuple(input_data)

# temp = predict_train_data('NQH', 'train')
# while(temp.next_data()):
#     temp.get_input_batch_data()
    # while(True):
    #     done, _, _ = temp.get_input_one_data()
    #     if done:
    #         break