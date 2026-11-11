from matplotlib import pyplot
from pyBat import Smoothn
import numpy


def plot_horizontal_lines(x, y):
    width = 0.8/2
    for x, y in zip(x, y):
        pyplot.hlines(y, x - width, x + width, colors='black', linestyles='solid', linewidth=3)  # Change color/style as desired

def set_face_color(color):
    ax = pyplot.gca()
    ax.set_facecolor(color)

def square_axis():
    axis = pyplot.gca()
    axis.set_aspect('equal', 'box')

def vlines(xs, **kwargs):
    ax = pyplot.gca()
    limit = ax.get_ylim()
    pyplot.vlines(xs, limit[0], limit[1], **kwargs)


def hlines(ys, **kwargs):
    ax = pyplot.gca()
    limit = ax.get_xlim()
    pyplot.hlines(ys, limit[0], limit[1], **kwargs)

def label(txt, x, y, c='black'):
    ax = pyplot.gca()
    pyplot.text(x, y, txt, color=c, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


def smooth_hist2d(x, y, bins=20, cmap='hot', smooth=1, levels=10, type='f'):

    result = numpy.histogram2d(x, y, bins=bins)
    xticks = result[1]
    xticks = xticks[0] + numpy.cumsum(numpy.diff(xticks))

    yticks = result[2]
    yticks = yticks[0] + numpy.cumsum(numpy.diff(yticks))

    surf = result[0]
    surf = surf.transpose()
    surf_smoothed = Smoothn.smoothn(surf, s= smooth)
    surf_smoothed = surf_smoothed[0]
    surf_smoothed[surf_smoothed<numpy.min(surf)] = numpy.min(surf)
    surf_smoothed[surf_smoothed>numpy.max(surf)] = numpy.max(surf)

    if type == 'f':
        pyplot.contourf(xticks, yticks, surf_smoothed, cmap=cmap, levels=levels)
    else:
        pyplot.contour(xticks, yticks, surf_smoothed, cmap=cmap, levels=levels)