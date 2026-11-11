from library import Masking
from pyBat import SaveLoad
import numpy
from pyBat import Bat
from pyBat import Acoustics
from os import path
import pandas
import copy

atmosphere = Acoustics.default_atmosphere()
sound_speed = atmosphere.soundspeed

parts = path.split(__file__)
folder = parts[0]
full_path = path.join(folder, 'stored_hrtfs.pck')
stored_hrtfs = SaveLoad.pickle_load(full_path)


def select_arena_reflectors(arena_reflectors, proportion):
    num_rows_to_select = int(proportion * arena_reflectors.shape[0])
    selected_indices = numpy.random.choice(arena_reflectors.shape[0], num_rows_to_select, replace=False)
    selected_rows = arena_reflectors[selected_indices]
    return selected_rows


def create_perched_bats(arena_reflectors, proportion = 0.1):
    selected_reflectors = select_arena_reflectors(arena_reflectors, proportion)
    rows = selected_reflectors.shape[0]
    bats = []
    for index in range(rows):
        x = selected_reflectors[index, 0]
        y = selected_reflectors[index, 1]
        z = selected_reflectors[index, 2]
        yaw = numpy.random.uniform(-180, 180)
        pitch = numpy.random.uniform(-90, 90)
        current = Bat.Bat(logger='simple')
        current.id = index
        current.goto(x=x, y=y, z=z, yaw=yaw, pitch=pitch, update_log=False)
        current = copy.copy(current)
        bats.append(current)
    return bats


def closest_point_per_segment(azimuths, distances, segment_size=10):
    result = numpy.zeros_like(distances, dtype=int)
    for segment_start in range(-180, 180, segment_size):
        segment_end = segment_start + segment_size
        mask = (azimuths >= segment_start) & (azimuths < segment_end)
        if numpy.any(mask):
            closest_idx = numpy.argmin(distances[mask])
            result[numpy.where(mask)[0][closest_idx]] = 1
    return result


def get_wall_spreading(shape):
    values = numpy.random.uniform(-20, -40, shape)
    return values


def get_hearing_spreading(shape):
    values = numpy.random.uniform(0, -20, shape)
    return values


def get_emission_spreading(shape):
    values = numpy.ones(shape) * -20
    return values


def get_pulses(focus_bat, other_bats, arena_reflectors, perched_bats, pulse_type, band="mid"):
    wall_ts = -12
    wall_ts_range = 3
    nose_function, left_function, right_function, attenuation_coef = stored_hrtfs[band]
    min_distance = numpy.nan  # not all types will generate a valid min distance
    #This min_distance is not used to simulate the acoustics. It's only used to determine the closest bat and wall point

    # The calls of others as heard by the focus bat
    if pulse_type == 'calls':
        relative_positions_others, ids = focus_bat.get_others_relative_positions(other_bats, spherical=True)
        relative_positions_self, _ = focus_bat.get_self_relative_position(other_bats, spherical=True)
        emission_az = relative_positions_self[:, 0]
        emission_el = relative_positions_self[:, 1]
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        distance = relative_positions_others[:, 2]
        distance[distance < 0.1] = 0.1

        emission = nose_function((emission_az, emission_el))
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))
        attenuation = -attenuation_coef * distance

        spreading = -20 * numpy.log10(distance)

        left = emission + hearing_left + spreading + attenuation
        right = emission + hearing_right + spreading + attenuation

    # The echoes of others' bodies as heard by the focus bat
    if pulse_type == 'echoes':
        relative_positions_others, _ = focus_bat.get_others_relative_positions(other_bats, spherical=True)
        emission_az = relative_positions_others[:, 0]
        emission_el = relative_positions_others[:, 1]
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        distance = relative_positions_others[:, 2]
        min_distance = numpy.min(distance)
        distance[distance < 0.1] = 0.1

        emission = nose_function((emission_az, emission_el))
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))
        attenuation = -attenuation_coef * 2 * distance
        spreading = -40 * numpy.log10(distance)
        target_strength = numpy.random.uniform(-44, -10, distance.shape)
        left = emission + hearing_left + spreading + attenuation + target_strength
        right = emission + hearing_right + spreading + attenuation + target_strength

    # The echoes of the walls as heard by the focus bat
    if pulse_type == 'walls':
        relative_positions_others = focus_bat.world2bat(arena_reflectors, spherical=True)
        emission_az = relative_positions_others[:, 0]
        emission_el = relative_positions_others[:, 1]
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        distance = relative_positions_others[:, 2]
        min_distance = numpy.min(distance)
        distance[distance < 0.1] = 0.1

        selected_wall_points = closest_point_per_segment(emission_az, distance)
        non_selected_wall_points = 1 - selected_wall_points
        masking = non_selected_wall_points * -1000

        emission = nose_function((emission_az, emission_el))
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))
        attenuation = -attenuation_coef * 2 * distance
        wall_spreading = get_wall_spreading(distance.shape)
        spreading = wall_spreading * numpy.log10(distance)

        target_strength = numpy.random.uniform(-wall_ts_range, wall_ts_range, distance.shape) + wall_ts #TARGET STRENGTH WALLS
        left = emission + hearing_left + spreading + attenuation + target_strength + masking
        right = emission + hearing_right + spreading + attenuation + target_strength + masking

    # The echoes of others' bodies, as a result of others calls, as heard by the focus bat
    if pulse_type == 'echoes2':
        relative_positions_others, _ = focus_bat.get_others_relative_positions(other_bats, spherical=True)
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        hearing_distance = relative_positions_others[:, 2]
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))

        attenuation = []
        left = []
        right = []
        distance = []

        for other_bat in other_bats:
            relative_positions_others, _ = other_bat.get_others_relative_positions(other_bats, spherical=True)
            emission_az = relative_positions_others[:, 0]
            emission_el = relative_positions_others[:, 1]
            emission_distance = relative_positions_others[:, 2]
            current_distance = emission_distance + hearing_distance
            current_distance[current_distance < 0.1] = 0.1
            emission = nose_function((emission_az, emission_el))

            current_attenuation = -attenuation_coef * current_distance
            spreading = -40 * numpy.log10(current_distance)
            target_strength = numpy.random.uniform(-44, -10, 1)

            current_left = emission + hearing_left + spreading + current_attenuation + target_strength
            current_right = emission + hearing_right + spreading + current_attenuation + target_strength

            attenuation.append(current_attenuation)
            left.append(current_left)
            right.append(current_right)
            distance.append(current_distance)

        attenuation = numpy.concatenate(attenuation)
        left = numpy.concatenate(left)
        right = numpy.concatenate(right)
        distance = numpy.concatenate(distance)

    # The echoes of walls, as a result of others calls, as heard by the focus bat
    if pulse_type == 'walls2':
        relative_positions_others = focus_bat.world2bat(arena_reflectors, spherical=True)
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        hearing_distance = relative_positions_others[:, 2]
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))

        hearing_distance[hearing_distance < 0.1] = 0.1
        hearing_spreading = get_hearing_spreading(hearing_distance.shape)
        hearing_spreading = hearing_spreading * numpy.log10(hearing_distance)

        selected_wall_points = closest_point_per_segment(hearing_az, hearing_distance)
        non_selected_wall_points = 1 - selected_wall_points
        masking = non_selected_wall_points * -1000

        attenuation = []
        left = []
        right = []
        distance = []

        for other_bat in other_bats:
            relative_positions_others = other_bat.world2bat(arena_reflectors, spherical=True)
            emission_az = relative_positions_others[:, 0]
            emission_el = relative_positions_others[:, 1]
            emission_distance = relative_positions_others[:, 2]

            current_distance = hearing_distance + emission_distance
            current_distance[current_distance < 0.1] = 0.1
            current_attenuation = -attenuation_coef * current_distance

            emission = nose_function((emission_az, emission_el))

            emission_distance[emission_distance < 0.1] = 0.1
            emission_spreading = get_emission_spreading(emission_distance.shape)
            emission_spreading = emission_spreading * numpy.log10(emission_distance)

            spreading = emission_spreading + hearing_spreading

            #spreading = get_wall_spreading(current_distance.shape)
            #spreading = spreading * numpy.log10(current_distance)

            target_strength = numpy.random.uniform(-wall_ts_range, wall_ts_range, current_distance.shape) + wall_ts #TARGET STRENGTH WALLS

            current_left = emission + hearing_left + spreading + current_attenuation + target_strength + masking
            current_right = emission + hearing_right + spreading + current_attenuation + target_strength + masking

            attenuation.append(current_attenuation)
            left.append(current_left)
            right.append(current_right)
            distance.append(current_distance)

        attenuation = numpy.concatenate(attenuation)
        left = numpy.concatenate(left)
        right = numpy.concatenate(right)
        distance = numpy.concatenate(distance)

    # The calls of others on walls as heard by the focus bat
    if pulse_type == 'perched':
        relative_positions_others, ids = focus_bat.get_others_relative_positions(perched_bats, spherical=True)
        relative_positions_self, _ = focus_bat.get_self_relative_position(perched_bats, spherical=True)
        emission_az = relative_positions_self[:, 0]
        emission_el = relative_positions_self[:, 1]
        hearing_az = relative_positions_others[:, 0]
        hearing_el = relative_positions_others[:, 1]
        distance = relative_positions_others[:, 2]
        distance[distance < 0.1] = 0.1

        emission = nose_function((emission_az, emission_el))
        hearing_left = left_function((hearing_az, hearing_el))
        hearing_right = right_function((hearing_az, hearing_el))
        attenuation = -attenuation_coef * distance

        spreading = -20 * numpy.log10(distance)

        selected_wall_points = closest_point_per_segment(hearing_az, distance)
        non_selected_wall_points = 1 - selected_wall_points
        masking = non_selected_wall_points * -1000

        left = emission + hearing_left + spreading + attenuation + masking
        right = emission + hearing_right + spreading + attenuation + masking



    summary = {}
    summary['min_distance'] = min_distance
    summary['pulse_type'] = pulse_type

    result = {}
    result['min_distance'] = min_distance
    result['pulse_type'] = pulse_type
    # result['max_distance'] = max_distance
    result['attenuation'] = attenuation
    result['left'] = left
    result['right'] = right
    result['summed'] = left + right
    result['diffed'] = right - left
    result['maxed'] = numpy.maximum(left, right)
    result['distance'] = distance
    result = pandas.DataFrame(result)
    return result, summary
