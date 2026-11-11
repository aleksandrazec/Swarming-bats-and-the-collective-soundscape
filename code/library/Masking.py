import numpy as np


def count_intersections(points, spheres, radius):
    intersections = np.zeros(len(points), dtype=int)
    for i, (x_p, y_p, z_p) in enumerate(points):
        A = x_p ** 2 + y_p ** 2 + z_p ** 2
        for (x_c, y_c, z_c) in spheres:
            B = -2 * (x_p * x_c + y_p * y_c + z_p * z_c)
            C = x_c ** 2 + y_c ** 2 + z_c ** 2 - radius ** 2
            discriminant = B ** 2 - 4 * A * C

            if discriminant >= 0:
                t1 = (-B + np.sqrt(discriminant)) / (2 * A)
                t2 = (-B - np.sqrt(discriminant)) / (2 * A)
                if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                    intersections[i] += 1
    return intersections


def masking_bodies(focus_bat, arena_reflectors, other_bats):
    relative_positions_walls = focus_bat.world2bat(arena_reflectors, spherical=False)
    relative_positions_others, _ = focus_bat.get_others_relative_positions(other_bats, spherical=False)
    count = count_intersections(relative_positions_walls, relative_positions_others, 0.14)
    # n = count_points_within_distance(relative_positions_walls, relative_positions_others, threshold)
    # n = np.array(n)
    return count


def distance_from_line(point, line_direction):
    # point is B_j, line_direction is A_i
    cross_product = np.cross(point, line_direction)
    distance = np.linalg.norm(cross_product) / np.linalg.norm(line_direction)
    return distance


def count_points_within_distance(set_A, set_B, threshold_distance):
    count_results = []

    for A_i in set_A:
        count = 0
        for B_j in set_B:
            distance = distance_from_line(B_j, A_i)
            if distance <= threshold_distance:
                count += 1
        count_results.append(count)

    return count_results



