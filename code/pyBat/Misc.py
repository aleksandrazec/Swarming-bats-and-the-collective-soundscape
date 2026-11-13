import os
import sys
import numpy
import datetime
from collections.abc import Iterable  # drop `.abc` with Python 2.7 or lower
import copy

# Trick to get basemap to work
path = sys.executable
path = path.replace('bin/python', '')
os.environ['PROJ_LIB'] = path + 'share/proj'
from mpl_toolkits.basemap import Basemap

def remove_from_list(list, indices):
    new_list = copy.copy(list)
    for index in sorted(indices, reverse=True):
        del new_list[index]
    return new_list


def find_y_in_x(x, y):
    xsorted = numpy.argsort(x)
    ypos = numpy.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices


def lst2str(lst, sep=','):
    s = ''
    for x in lst: s += str(x) + sep
    s = s.rstrip(sep)
    return s


def get_date_time():
    time_object = datetime.datetime.now()
    day = time_object.day
    month = time_object.month
    year = time_object.year
    hour = time_object.hour
    minute = time_object.minute
    second = time_object.second
    lst = [day, month, year, hour, minute, second]
    return lst


def inrange(x, min_value, max_value=None):
    if max_value is None:
        max_value = min_value[1]
        min_value = min_value[0]
    if x < max_value and x > min_value: return True
    return False


def list2lower(lst):
    result = [x.lower() for x in lst]
    return result


def plot_map(az, el, z, levels, cmap='inferno'):
    mmap = Basemap(projection='hammer', lat_0=0, lon_0=0)
    x, y = mmap(az, el)
    mmap.contourf(x, y, z, tri=True, cmap=cmap, levels=levels)
    parallels = numpy.arange(-90, 90, 60)
    mmap.drawparallels(parallels)
    meridians = numpy.arange(-180, 180, 60)
    mmap.drawmeridians(meridians)

def mat2array(matrix):
    array = numpy.array(matrix, dtype='f')
    array = numpy.squeeze(array)
    if array.shape == (): array = numpy.reshape(array, (1,))
    return array


def normalize_vector(vector):
    norm = numpy.linalg.norm(vector)
    if norm == 0: return vector
    x = vector / norm
    return x


def angle_between(v1, v2):
    v1 = numpy.reshape(v1, (1, -1))
    v2 = numpy.reshape(v2, (-1, 1))
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    angle = numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))
    angle = float(angle)
    angle = numpy.rad2deg(angle)
    if numpy.isnan(angle): return 90
    return angle

def signed_vector_angle(p1, p2):
    ang1 = numpy.arctan2(*p1[::-1])
    ang2 = numpy.arctan2(*p2[::-1])
    return numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))


def inter_distance(start1, end1, start2, end2):
    start1 = numpy.array(start1, dtype='f')
    start2 = numpy.array(start2, dtype='f')

    end1 = numpy.array(end1, dtype='f')
    end2 = numpy.array(end2, dtype='f')

    norm1 = numpy.linalg.norm(end1 - start1)
    norm2 = numpy.linalg.norm(end2 - start2)

    n = (numpy.ceil(max(norm1, norm2)) + 1) * 2

    x1 = numpy.linspace(start1[0], end1[0], n)
    y1 = numpy.linspace(start1[1], end1[1], n)
    z1 = numpy.linspace(start1[2], end1[2], n)

    x2 = numpy.linspace(start2[0], end2[0], n)
    y2 = numpy.linspace(start2[1], end2[1], n)
    z2 = numpy.linspace(start2[2], end2[2], n)

    distance = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    distance = numpy.sqrt(distance)
    distance = numpy.min(distance)
    return distance


def append2csv(data, path, sep=","):
    import os
    if not os.path.isfile(path):
        data.to_csv(path, mode='a', index=False, sep=sep)
    else:
        data.to_csv(path, mode='a', index=False, sep=sep, header=False)


def scale_ranges(a, b, zoom_out=1.25):
    rng_a = numpy.ptp(a)
    rng_b = numpy.ptp(b)
    mean_a = numpy.mean(a)
    mean_b = numpy.mean(b)
    if rng_b > rng_a: a = (b - mean_b) + mean_a
    if rng_a > rng_b: b = (a - mean_a) + mean_b
    mean_a = numpy.mean(a)
    mean_b = numpy.mean(b)
    a = ((a - mean_a) * zoom_out) + mean_a
    b = ((b - mean_b) * zoom_out) + mean_b
    return a, b


def minmax(array, expand_factor=0, expand=0):
    mn = numpy.nanmin(array)
    delta_mn = numpy.abs(mn) * expand_factor
    mx = numpy.nanmax(array)
    delta_mx = numpy.abs(mx) * expand_factor
    mn = (mn - delta_mn) - expand
    mx = (mx + delta_mx) + expand
    rng = numpy.array((mn, mx), dtype='f')
    return rng


def unwrap(angles):
    radians = numpy.deg2rad(angles)
    radians = numpy.unwrap(radians)
    angels = numpy.rad2deg(radians)
    return angels


def iterable(obj, allow_string=False):
    if isstr(obj) and not allow_string: return False
    return isinstance(obj, Iterable)


def isstr(x):
    return isinstance(x, str)


def nan_array(shape):
    return numpy.full(shape, numpy.nan)


def rand_range(min_value, max_value, shape):
    y = numpy.random.random(shape)
    y = y * (max_value - min_value)
    y = y + min_value
    return y


def unit_vector(norm=1):
    return numpy.array([[0], [0], [1]], dtype='f') * norm


def closest(array, value):
    idx = (numpy.abs(array - value)).argmin()
    return array[idx], idx


def angle_arrays(az_range=180, el_range=90, step=2.5, grid=True):
    az_range = abs(az_range)
    el_range = abs(el_range)
    az = numpy.arange(-az_range, az_range + 0.001, step)
    az = numpy.transpose(az)
    el = numpy.arange(-el_range, el_range + 0.001, step)
    if not grid: return az, el
    az, el = numpy.meshgrid(az, el)
    return az, el


def random_cds(n, min_value, max_value, y_zero=True):
    points = numpy.random.rand(n, 3)
    r = max_value - min_value
    points = (points * r) + min_value
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    if y_zero: y = numpy.zeros(x.shape)
    return x, y, z


def almost_equal(a, b, threshold):
    diff = abs(a - b)
    if diff <= threshold: return True
    return False


def sign(x):
    if x < 0: return -1
    if x > 0: return 1
    if x == 0: return 0


def values2labels(lst):
    labels = []
    for x in lst:
        label = 'Condition ' + str(x + 1)
        labels.append(label)
    return labels
    #    if x == 0:
    #        labels.append('Fixed')
    #    else:
    #        labels.append(str(x) + ' deg/s$^2$')
    # return labels


def smooth(x, window_len=11, window='hanning'):
    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = eval('signal.windows.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def zero_pad(array, length, value=0):
    n = len(array)
    if n > length: return array[:length]
    padding = length - n
    new = numpy.pad(array, (0, padding), mode='constant', constant_values=value)
    return new



