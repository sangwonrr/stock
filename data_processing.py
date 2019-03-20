import numpy as np
import os
import re

class preprocessing_tick():
    def __init__(self, code):
        self.code = code
        self.tict_data_dir = './tick_data'
        self.collected_data = {}
        self.hour_interval = 2

    def get_output_dir(self):
        return f'./data/{self.code}'

    def read_file(self):
        regex = re.compile(f'tick_\S*{self.code}\S*.npy')
        for root, dirs, files in os.walk(self.tict_data_dir):
            for fname in files:
                if regex.search(fname) == None:
                    continue
                reading_data = np.load(os.path.join(root, fname))
                self.collect_data(reading_data)

    def collect_data(self, reading_data):
        #date time price volume
        for idx in range(len(reading_data)):
            temp_data = reading_data[idx]
            date = int(temp_data[0])
            time = int(temp_data[1])
            if self.collected_data.get(date) == None:
                self.collected_data[date] = {}
                self.collected_data[date][time] = [temp_data[2], int(temp_data[3])]
                continue
            if self.collected_data[date].get(time) == None:
                self.collected_data[date][time] = [temp_data[2], int(temp_data[3])]
                continue
            self.collected_data[date][time][0] = temp_data[2]
            self.collected_data[date][time][1] += int(temp_data[3])

    def export_data(self):
        key_interval = self.hour_interval * 100
        idx_interval = self.hour_interval * 60
        for date_key in self.collected_data.keys():
            time_keys = list(self.collected_data[date_key].keys())
            time_keys.sort()
            candidate = {}
            # split by 2 hours ( 0200 ~ 0359 ) because 0 is next day
            for idx, time_key in zip(range(len(time_keys)), time_keys):
                if time_key % key_interval == 0:
                    candidate[time_key] = idx

            for key, idx in candidate.items():
                nextkey = key + key_interval
                if key == (2400 - key_interval):
                    if idx_interval != (len(time_keys) - idx):
                        continue
                elif candidate.get(nextkey) == None or (candidate[nextkey] - idx) != idx_interval:
                    continue

                final_data = []
                for interval in range(self.hour_interval):
                    start = key + 100 * interval
                    end = start + 60
                    for time_idx in range(start, end):
                        time_data = []
                        time_data.append(time_idx)
                        time_data.append(self.collected_data[date_key][time_idx][0])
                        time_data.append(self.collected_data[date_key][time_idx][1])
                        final_data.append(time_data)
                self._save_export_data(final_data, date_key, key)

    def _save_export_data(self, final_data, date, time):
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.get_output_dir())
        if os.path.lexists(output_dir) == False:
            os.makedirs(output_dir)
        file_name = f'{self.code}_{date}_{time}'
        full_path = os.path.join(output_dir, file_name)
        np.save(full_path, final_data)



# test = preprocessing_tick('ESH')
# test.read_file()
# test.export_data()