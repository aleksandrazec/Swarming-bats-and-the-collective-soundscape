import numpy
import pandas
from pyBat import Misc


class SimpleLogger:
    def __init__(self):
        self.data = {}

    @property
    def keys(self):
        return self.data.keys()

    @property
    def time(self):
        if 'time' in self.keys:
            time = self.data['time']
            time = numpy.array(time, dtype='f')
            return time
        else:
            return None

    def __getitem__(self, field):
        return self.get(field)

    def __setitem__(self, field, data):
        if not Misc.iterable(field, allow_string=False): field = [field]
        if not Misc.iterable(data): data = [data]
        for field_i, data_i in zip(field, data):
            keys = self.keys
            if field_i in keys:
                series = self.data[field_i]
                series.append(data_i)
                self.data[field_i] = series
            if field_i not in keys:
                self.data[field_i] = [data_i]

    def get(self, field):
        data = self.data[field]
        data = numpy.array(data)
        return data

    def export(self):
        df = pandas.DataFrame(self.data)
        return df

    def print_lengths(self):
        keys = self.data.keys()
        for k in keys: print(k, len(self.data[k]))


if __name__ == "__main__":
    import random

    log = SimpleLogger()
    for i in range(10):
        log['time'] = i
        log['test1'] = random.random()
        log['test2'] = random.random()
        log['cat'] = 'ABC'
    export = log.export()
