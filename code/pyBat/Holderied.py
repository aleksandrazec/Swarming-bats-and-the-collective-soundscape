import numpy

def gforce(velocity, turning_rate):
    rate = numpy.deg2rad(turning_rate)
    g = (velocity * rate) / 9.81
    return g


def turning_rate(velocity, gforce=2):
    rate = (gforce * 9.81) / velocity
    rate = numpy.rad2deg(rate)
    return rate


