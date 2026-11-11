import copy

import numpy
from matplotlib import pyplot
from skimage import feature
from skimage.io import imread
from skimage.morphology import opening, disk, closing
from skimage.morphology import skeletonize
from shapely import geometry

from library import SortPoints
from library import Utils
import cairosvg
from os.path import join, exists

arena_path = join('..', 'arenas')
path_exists = exists(arena_path)
if not path_exists:
    arena_path = join('.', 'arenas')


def read_arena(base_name, scale, plot=False):
    footprint = disk(4)
    full_svg = join(arena_path, base_name + '.svg')
    full_png = join(arena_path, base_name + '.png')
    cairosvg.svg2png(url=full_svg, write_to=full_png, output_width=500, output_height=500)
    image = imread(full_png)

    grey = numpy.maximum(image[:, :, 0] , image[:, :, 1])
    grey = numpy.maximum(grey, image[:, :, 2])
    grey = (grey > 100) * 255

    path = grey > 10

    start = (image[:, :, 1] > 100) * (image[:, :, 2] < 100)
    end = (image[:, :, 1] < 100) * (image[:, :, 2] > 100)

    path = numpy.maximum(path, start)

    start_surface = numpy.sum(start) * (scale ** 2)

    path = opening(path, footprint)
    start = opening(start, footprint)

    skeleton = skeletonize(path)

    path_edge = feature.canny(path, sigma=3)

    if plot:
        pyplot.figure(figsize=(10,5))
        pyplot.subplot(2, 3, 1)
        pyplot.imshow(path, cmap='gray')
        pyplot.title('Path')
        pyplot.subplot(2, 3, 2)
        pyplot.imshow(start, cmap='gray')
        pyplot.title('Start')
        pyplot.subplot(2, 3, 3)
        pyplot.imshow(end, cmap='gray')
        pyplot.title('End')

        pyplot.subplot(2, 3, 4)
        pyplot.imshow(path_edge, cmap='gray')
        pyplot.title('Path Edge')

        pyplot.subplot(2, 3, 5)
        pyplot.imshow(skeleton, cmap='gray')
        pyplot.title('Skeleton')

        pyplot.tight_layout()
        pyplot.show()

    path_rows, path_cols = path_edge.nonzero()
    start_rows, start_cols = start.nonzero()
    end_rows, end_cols = end.nonzero()
    center_rows, center_cols = skeleton.nonzero()

    dx = numpy.mean(path_cols)
    dy = numpy.mean(path_rows)

    path_x_cds = (path_cols - dx) * scale
    path_y_cds = -(path_rows - dy) * scale
    path_z_cds = Utils.symmetric_uniform(path_x_cds, 1)

    start_x_cds = (start_cols - dx) * scale
    start_y_cds = -(start_rows - dy) * scale
    start_z_cds = start_x_cds * 0

    end_x_cds = (end_cols - dx) * scale
    end_y_cds = -(end_rows - dy) * scale
    end_z_cds = end_x_cds * 0

    center_x_cds = (center_cols - dx) * scale
    center_y_cds = -(center_rows - dy) * scale
    center_z_cds = center_x_cds * 0

    if plot:
        pyplot.figure()
        pyplot.scatter(path_x_cds, path_y_cds)
        pyplot.scatter(start_x_cds, start_y_cds)
        pyplot.scatter(end_x_cds, end_y_cds)
        pyplot.scatter(center_x_cds, center_y_cds)
        pyplot.xlabel('m')
        pyplot.ylabel('m')
        ax = pyplot.gca()
        ax.set_aspect(1)
        pyplot.tight_layout()
        pyplot.show()


    if base_name == 'open':
        selected_x = path_x_cds < 2
        path_x_cds = path_x_cds[selected_x]
        path_y_cds = path_y_cds[selected_x]
        path_z_cds = path_z_cds[selected_x]


    edge_points = numpy.column_stack((path_x_cds, path_y_cds, path_z_cds))
    sorted_edge_points = SortPoints.sortPoints(edge_points)
    edge_poly = geometry.Polygon(sorted_edge_points)


    if plot:
        pyplot.figure()
        edges_x = sorted_edge_points[:-1, 0]
        edges_y = sorted_edge_points[:-1, 1]
        pyplot.show()
        pyplot.figure()
        pyplot.scatter(edges_x, edges_y, color='r', alpha=0.5)
        pyplot.plot(edges_x, edges_y, color='k', alpha=0.5)
        pyplot.show()
        pyplot.figure()
        pyplot.plot(*edge_poly.exterior.xy)
        pyplot.show()

    result = {}
    result['edge_points'] = edge_points
    result['sorted_edge_points'] = sorted_edge_points
    result['edge_poly'] = edge_poly
    result['start_region'] = numpy.column_stack((start_x_cds, start_y_cds, start_z_cds))
    result['end_region'] = numpy.column_stack((end_x_cds, end_y_cds, end_z_cds))
    result['center_points'] = numpy.column_stack((center_x_cds, center_y_cds, center_z_cds))
    #result['threshold_distance'] = max_distance
    result['start_image'] = start
    #result['start_radius'] = start_radius
    result['start_surface'] = start_surface
    result['base_name'] = base_name

    return result


def check_within(arena, point, reference='edge'):
    if reference == 'edge': poly = arena['edge_poly']
    point = geometry.Point([point[0], point[1]])
    result = point.within(poly)
    return result

if __name__ == "__main__":
    base_name = 'branched'
    r = read_arena(base_name, scale=0.1, plot=True)






