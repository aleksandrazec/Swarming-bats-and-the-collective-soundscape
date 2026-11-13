import numpy
from scipy.signal import freqz
from scipy.signal import gammatone, lfilter

from pyBat import Acoustics
from pyBat.Signal import LowPassFilter, rectify
from pyBat.Acoustics import db2ratio, ratio2db

try:
    from Gammatone import GammatoneFilterbank
except:
    from .Gammatone import GammatoneFilterbank

from matplotlib import pyplot


class Attenuation:
    def __init__(self, frequencies):
        self.atmosphere = Acoustics.default_atmosphere()
        self.frequencies = frequencies
        self.reflection_parameter = -40
        self.spreading_parameter = -40
        self.matrix = False

    def get_absorption(self, distance):  # in db
        absorption_coefficients = []
        for frequency in self.frequencies:
            coefficient = self.atmosphere.attenuation_coefficient(frequency) * -1
            absorption_coefficients.append(coefficient)
        absorption_coefficients = numpy.array(absorption_coefficients)
        absorption = absorption_coefficients * distance * 2
        return absorption

    def get_spreading(self, distance):  # in db
        spreading = self.spreading_parameter * numpy.log10(distance)
        return spreading

    def get_attenuation(self, distance, db=False):
        spreading = self.get_spreading(distance)
        absorption = self.get_absorption(distance)

        spreading = db2ratio(spreading)
        absorption = db2ratio(absorption)
        reflection = db2ratio(self.reflection_parameter)
        attenuation = spreading * absorption * reflection
        if db: attenuation = ratio2db(attenuation)
        return attenuation

    def attenuate_matrix(self, matrix, distance):
        columns = matrix.shape[1]
        attenuation = self.get_attenuation(distance, db=False)
        attenuation = attenuation.reshape(-1, 1)
        attenuation = numpy.repeat(attenuation, columns, axis=1)
        self.matrix = attenuation
        new_matrix = matrix * attenuation
        return new_matrix

class ModelWiegrebe:
    def __init__(self, sample_rate, center, bands):
        self.sample_rate = sample_rate
        self.center = center
        self.bands = bands
        self.gammatone = GammatoneFilterbank(samplerate=sample_rate, startband=-bands, endband=bands, normfreq=center, desired_delay_sec=0.001)
        if numpy.max(self.frequencies) > (sample_rate / 2) * 0.9:
            raise ValueError('Max frequency too high')
        self.signal = None
        self.filter = LowPassFilter(1000, sample_rate, 2)
        self.gamma_output = None
        self.rectified = None
        self.compressed = None
        self.filtered = None
        self.matrix = None
        self.result = None
        self.exponent = 0.4
        self.attenuation = Attenuation(frequencies=self.frequencies)

    @property
    def frequencies(self):
        return self.gammatone.centerfrequencies

    def set_lowpass(self, frequency):
        self.filter = LowPassFilter(frequency, self.sample_rate, 2)

    def run_gammatone(self):
        analyse = self.gammatone.analyze(self.signal)
        n = self.signal.size
        bm = numpy.zeros((0, n))
        for band in analyse:
            response = numpy.real(band[0])
            bm = numpy.vstack((bm, response))
        return bm

    def run_model(self, signal, attenuation=False, distance=0):
        self.signal = signal
        self.gamma_output = self.run_gammatone()
        if attenuation: self.gamma_output = self.attenuation.attenuate_matrix(self.gamma_output, distance)
        self.rectified = rectify(self.gamma_output)
        self.compressed = numpy.power(self.rectified, self.exponent)
        self.filtered = self.filter.run(self.compressed)
        self.matrix = rectify(self.filtered)
        self.result = numpy.mean(self.matrix, axis=0)
        return self.result

    def plot_filter(self):
        w, h = freqz(self.filter.b, self.filter.a)
        f = 0.5 * self.sample_rate * w / numpy.pi
        db = 20 * numpy.log10(abs(h))
        pyplot.plot(f, db)

    @property
    def time_axis(self):
        matrix = self.matrix
        max_time = (1 / self.sample_rate) * matrix.shape[1]
        x = numpy.linspace(0, 1, matrix.shape[1]) * max_time
        return x



    # def plot_matrix(self):
    #
    #     pyplot.contourf(x, y, matrix)
    #     pyplot.show()

#
#
# class ModelWiegrebeSingleFreq:
#     def __init__(self, center, sample_rate):
#         self.signal = None
#         self.gamma_output = None
#         self.rectified = None
#         self.compressed = None
#         self.filtered = None
#         self.result = None
#
#         self.exponent = 0.4
#         self.center_freq = center
#         self.fs = sample_rate
#         self.filter = LowPassFilter(1000, self.fs, 2)
#         # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gammatone.html
#         # This order of a and seems correct - see page above
#         self.b, self.a = gammatone(self.center_freq, 'fir', fs=self.fs)
#
#     def run_gammatone(self):
#         gammatone_result = lfilter(self.b, self.a, self.signal)
#         return gammatone_result
#
#     def run(self, signal):
#         self.signal = signal
#         self.gamma_output = self.run_gammatone()
#         self.rectified = rectify(self.gamma_output)
#         self.compressed = numpy.power(self.rectified, self.exponent)
#         self.filtered = self.filter.run(self.compressed)
#         self.result = rectify(self.filtered)
#         return self.result