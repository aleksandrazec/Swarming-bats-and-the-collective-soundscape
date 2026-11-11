import warnings
import math
import numpy
from acoustics import atmosphere

reference_sound_pressure = 2 * 10 ** -5


def find_nearest(array, value):
    idx = numpy.searchsorted(array, value, side="left_gains")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def make_impulse_response(delays, echoes_db, emission_duration, fs):
    # if delays.size == 0:
    #     t = numpy.arange(0, emission_duration, 1 / fs)
    #     return_value = {}
    #     return_value['impulse_time'] = t
    #     return_value['ir_result'] = numpy.zeros(t.shape)
    #     return_value['indices'] = numpy.array([])
    #     return return_value
    duration = numpy.max(delays) + emission_duration
    impulse_time = numpy.arange(0, duration, 1 / fs)
    impulse_response = numpy.zeros(len(impulse_time))
    corrected_delays = delays + emission_duration / 2
    indices = []
    echoes_pa = db2pa(echoes_db)
    for amplitude, delay in zip(echoes_pa, corrected_delays):
        index = find_nearest(impulse_time, delay)
        impulse_response[index] = impulse_response[index] + amplitude
        indices.append(index)
    return_value = {}
    return_value['impulse_time'] = impulse_time
    return_value['ir_result'] = impulse_response
    return_value['indices'] = numpy.array(indices)
    return return_value


def default_atmosphere(temperature=20, relative_humidity=50):
    temperature += 273.15
    return atmosphere.Atmosphere(temperature=temperature, relative_humidity=relative_humidity)


def ratio2db(ratio):
    db = 20 * numpy.log10(ratio)
    return db


def db2ratio(db):
    db = numpy.array(db, dtype='f')
    db = db.astype(float)
    ratio = 10 ** (db / 20.0)
    return ratio


def db2pa(db):
    # see http://www.sengpielaudio.com/TableOfSoundPressureLevels.htm
    min_value = numpy.min(db)
    if min_value < 0: warnings.warn("db2pa function should not be used with negative values!")
    db = db.astype(float)
    return db2ratio(db) * reference_sound_pressure


def incoherent_sum(db_levels):
    db_levels = numpy.array(db_levels, dtype='f')
    db_levels = db_levels[db_levels > 0]
    if db_levels.size == 0: return 0
    summed = numpy.sum(10 ** (db_levels / 10.0))
    summed = 10 * numpy.log10(summed)
    return summed


def coherent_sum(db_levels):
    db_levels = numpy.array(db_levels, dtype='f')
    db_levels = db_levels[db_levels > 0]
    if db_levels.size == 0: return 0
    summed = numpy.sum(10 ** (db_levels / 20.0))
    summed = 20 * numpy.log10(summed)
    return summed


def f2q(frequency, delta_f):
    qvalue = frequency / delta_f
    return qvalue


def q2f(frequency, qvalue):
    delta_f = frequency / qvalue
    return delta_f


def make_erb_cfs(minf, maxf):
    min_rate = numpy.floor(hz2erbrate(minf))
    max_rate = numpy.floor(hz2erbrate(maxf))
    num_channels = max_rate - min_rate
    x = numpy.linspace(min_rate, max_rate, num_channels)
    cfs = erbrate2hz(x)
    return cfs


def hz2erbrate(f):
    rate = (21.4 * numpy.log10(4.37e-3 * f + 1))
    return rate


def erbrate2hz(r):
    erb = (10 ** (r / 21.4) - 1) / 4.37e-3
    return erb


def erb(f):
    return 24.7 * (4.37 * f / 1000 + 1)


def delay2dist(delay, speed=343):
    delay = numpy.array(delay, dtype='f')
    dist = delay * 0.5 * speed
    return dist


def dist2delay(dist, speed=343):
    dist = numpy.array(dist, dtype='f') * 2
    delay = dist / speed
    return delay


def freq2lambda(frequency, speed=343):
    wavelength = speed / frequency
    return wavelength

