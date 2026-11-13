import math
import numpy
import pandas
from pyBat import Misc
from matplotlib import pyplot
from natsort import natsorted
from scipy.interpolate import interp1d

class Logger:
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
        if not Misc.iterable(field): field = [field]
        if not Misc.iterable(data): data = [data]
        for field_i, data_i in zip(field, data):
            data_i = float(data_i)
            keys = self.keys
            if field_i in keys:
                series = self.data[field_i]
                series.append(data_i)
                self.data[field_i] = series
            if field_i not in keys:
                self.data[field_i] = [data_i]

    def get(self, field, unwrap=False):
        data = self.data[field]
        data = numpy.array(data)
        if unwrap: data = Misc.unwrap(data)
        return data

    def get_history(self, field, time_stamps=None, gradient=False, unwrap=False, return_time=False, absolute=False):
        # Get the time axis for the field
        time = self.time
        if time is None: return numpy.zeros(1)
        # get time stamps
        if time_stamps is None: time_stamps = max(time)
        if isinstance(time_stamps, str) and time_stamps == 'all': time_stamps = time
        if not Misc.iterable(time_stamps): time_stamps = [time_stamps]
        time_stamps = numpy.array(time_stamps, dtype='f')
        # Get data
        keys = self.keys
        if field not in keys: return numpy.zeros(time_stamps.shape)
        data = self.get(field, unwrap)
        if time is None: time = range(0, len(data))
        # Make data interpolatable
        if len(data) < 2:
            data = numpy.ones(2) * data
            time = numpy.ones(2) * time
        # Apply abs
        if absolute: data = numpy.abs(data)
        # handle negative time stamps
        time_stamps[time_stamps < 0] = time_stamps[time_stamps < 0] + max(time)
        # handle time stamps not in range
        time_stamps[time_stamps < min(time)] = min(time)
        time_stamps[time_stamps > max(time)] = max(time)
        # get gradient
        if gradient:
            differentiable = numpy.max(numpy.abs(numpy.diff(data)))
            if differentiable > 0: data = numpy.gradient(data, time, edge_order=1)
            if differentiable == 0: data = numpy.zeros(data.shape)
            data[~numpy.isfinite(data)] = 0
        # interpolate
        first_value = data[0]
        last_value = data[-1]
        history_function = interp1d(time, data, fill_value=(first_value, last_value), bounds_error=False)
        values = history_function(time_stamps)
        if return_time: return time_stamps, values
        return values

    def export(self, time_stamps='all'):
        keys = self.keys
        keys = natsorted(keys)
        time_stamps = self.get_history(field='time', time_stamps=time_stamps)
        d = {'time': pandas.Series(time_stamps)}
        df = pandas.DataFrame(d)
        for key in keys:
            data = self.get_history(field=key, time_stamps=time_stamps)
            df[key] = data
        return df

    def export_to_html(self, time_stamps='all'):
        df = self.export(time_stamps)
        html = df.to_html()
        return html

    def plot(self, keys=None):
        time = self.time
        if not Misc.iterable(keys): keys = [keys]
        if keys is None: keys = list(self.keys)
        number = len(keys)
        columns = 2
        rows = math.ceil(number / columns)
        pyplot.figure()
        for index in range(0, len(keys)):
            current_key = keys[index]
            data = self.get(current_key)
            if time is None: time = range(0, len(data))
            if len(keys) > 1: pyplot.subplot(rows, columns, index + 1)
            pyplot.plot(time, data, '.-')
            pyplot.xlabel('time')
            pyplot.ylabel(current_key)
            pyplot.title(current_key)
        pyplot.tight_layout()


if __name__ == "__main__":
    import random
    log = Logger()
    for i in range(10):
        log['time'] = i * 1.1
        log['test1'] = random.random()
        log['test2'] = random.random()
    export = log.export('all')