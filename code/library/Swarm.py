import copy
import random
import warnings

import numpy
import pandas
from matplotlib import pyplot
from sklearn.metrics import pairwise_distances

from library import PulseGeneration
from library import ReadArena
from library import Tadarida

from pyBat import Bat
from pyBat import Geometry
from pyBat import Holderied
from pyBat import Misc

from library import PaperSettings
from library import Utils

half_width = 125  # half width of integration window


class Swarm:
    def __init__(self, nr_of_bats, arena, name='swarm', condition='default', replication=None):
        self.max_speed = 9
        self.pulse_interval = 1/25
        self.processing_interval = self.pulse_interval * 0.5 # This is the section of the signal the bats actually process

        self.name = str(name)
        self.replication = replication
        self.condition = 0
        self.simulation_step_nr = 0

        self.active_bats = []
        self.inactive_bats = []
        self.all_pulse_data = []
        self.condition = condition
        self.perched_bats = []
        self.outside_positions = []

        self.arena = arena
        self.arena_type = arena['base_name']
        self.start_region = arena['start_region']
        self.end_region = arena['end_region']
        self.edge_points = arena['edge_points']
        self.center_points = arena['center_points']
        self.wall_collisions = 0

        n_start_positions = self.start_region.shape[0]

        for index in range(nr_of_bats):
            start_position = random.randint(0, n_start_positions - 1)
            start_position = self.start_region[start_position, :]

            random_x = start_position[0]
            random_y = start_position[1]
            random_z = numpy.random.uniform(-0.5, 0.5, 1)[0]
            random_yaw = numpy.random.uniform(-10, 10, 1)

            current = Bat.Bat(logger='simple')
            current.speed = self.max_speed
            current.id = index
            current.goto(x=random_x, y=random_y, z=random_z, yaw=random_yaw[0], update_log=False)
            current = copy.copy(current)
            self.active_bats.append(current)

        results_folder = PaperSettings.results_folder
        self.output_folder = Utils.get_last_part_of_path(results_folder)

    def __getitem__(self, index):
        return self.active_bats[index]

    @property
    def nr_of_bats(self):
        return len(self.active_bats)

    @property
    def all_bats(self):
        return self.active_bats + self.inactive_bats

    @property
    def bat_positions(self):
        positions = []
        for bat in self.active_bats: positions.append(bat.position)
        positions = numpy.array(positions)
        return positions

    def set_perched_bats(self, proportion=0.5):
        arena_reflectors = self.edge_points
        perched = PulseGeneration.create_perched_bats(arena_reflectors, proportion)
        self.perched_bats = perched

    def in_arena(self, current_position):
        return ReadArena.check_within(self.arena, current_position, reference='edge')

    def get_all_pulse_data(self):
        return pandas.concat(self.all_pulse_data)

    def get_bat_distances(self, matrix=False):
        positions = self.bat_positions
        distances = pairwise_distances(positions)
        if matrix: return distances
        selected = numpy.triu(distances, k=1)
        selected = selected[selected > 0]
        return selected

    def get_closest_distances(self):
        matrix = self.get_bat_distances(matrix=True)
        matrix[matrix == 0] = numpy.nan
        closest = numpy.nanmin(matrix, axis=1)
        return closest

    def remove_bats_in_end_region(self, threshold=0.15, remove_bat=True):
        result = []
        indices_to_remove = []
        for i, current in enumerate(self.active_bats):
            distance, _ = self.get_closest_point(i, type='end')
            in_region = distance < threshold
            result.append(in_region)
            if in_region: indices_to_remove.append(i)
        result = numpy.array(result)
        for i in indices_to_remove:
            copied = copy.copy(self.active_bats[i])
            self.inactive_bats.append(copied)
        self.active_bats = Misc.remove_from_list(self.active_bats, indices_to_remove)
        return result

    def get_closest_point(self, bat_nr, type, ignore_z=True):
        if type == 'edge': points = self.edge_points
        if type == 'end': points = self.end_region
        if type == 'center': points = self.center_points
        current = self.active_bats[bat_nr]
        c_x = points[:, 0]
        c_y = points[:, 1]
        c_z = points[:, 2]
        x = current.x
        y = current.y
        z = current.z

        if ignore_z:
            z = z * 0
            c_z = c_z * 0

        distances = ((c_x - x) ** 2 + (c_y - y) ** 2 + (c_z - z) ** 2) ** 0.5
        index = numpy.argmin(distances)
        point = points[index, :]
        if ignore_z: point[2] = 0
        distance = distances[index]
        return distance, point

    def plot_bat_positions(self, view='top'):
        pyplot.figure()
        positions = self.bat_positions
        if view == 'top': pyplot.scatter(positions[:, 0], positions[:, 1])
        if view == 'side': pyplot.scatter(positions[:, 0], positions[:, 2])
        ax = pyplot.gca()
        ax.set_aspect(1)
        pyplot.show()

    def plot_traces(self, field='pulse_type_code', start_index=0, end_index=-1):
        pyplot.figure()
        all_bats = self.all_bats
        for current_bat in all_bats:
            x_path, y_path, _ = current_bat.get_path()
            data = current_bat.get_history(field)
            if len(x_path) > 2:
                x_path = x_path[start_index: end_index]
                y_path = y_path[start_index: end_index]
                data = data[start_index: end_index]

            pyplot.plot(x_path, y_path, alpha=0.25, linestyle='-', color='black')
            if field == 'pulse_type_code':
                pyplot.scatter(x_path, y_path, c=data, s=10, cmap='Set1', vmin=0, vmax=4)
                pyplot.title('0: echoes, 1: walls, 2: calls, 3: echoes2, 4: walls2')
            else:
                pyplot.scatter(x_path, y_path, c=data, s=10, cmap='viridis', vmin=0, vmax=1)

        pyplot.colorbar()
        x_edge = self.edge_points[:, 0]
        y_edge = self.edge_points[:, 1]
        pyplot.scatter(x_edge, y_edge)
        ax = pyplot.gca()
        ax.set_aspect(1)
        pyplot.show()

    def get_pulses_single_bat(self, index):
        focus_bat = self.active_bats[index]
        all_other_bats = copy.copy(self.active_bats)
        all_other_bats.pop(index)
        arena_reflectors = self.edge_points
        perched_bats = self.perched_bats
        condition = self.condition

        if condition == 'muted':
            focus_bat_calls = index % 2 == self.simulation_step_nr % 2
            if self.simulation_step_nr % 2 == 0:
                all_other_bats = [all_other_bats[i] for i in range(len(all_other_bats)) if i % 2 == 0]
            else:
                all_other_bats = [all_other_bats[i] for i in range(len(all_other_bats)) if i % 2 == 1]

        echoes, echoes_summary = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'echoes')
        calls, calls_summary = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'calls')
        walls, walls_summary = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'walls')
        echoes2, echoes_summary2 = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'echoes2')
        walls2, walls_summary2 = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'walls2')
        perched, perched_summary = PulseGeneration.get_pulses(focus_bat, all_other_bats, arena_reflectors, perched_bats, 'perched')

        if condition == 'muted':  # This condition mutes some of the bats for each iteration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                if focus_bat_calls:
                    all_pulses1 = pandas.concat((echoes, walls, calls, echoes2, walls2), ignore_index=True)
                    all_summary1 = pandas.DataFrame.from_records((echoes_summary, calls_summary, walls_summary, echoes_summary2, walls_summary2))
                    all_pulses2 = copy.copy(all_pulses1)
                    all_summary2 = copy.copy(all_summary1)
                else:
                    all_pulses1 = pandas.concat((calls, echoes2, walls2), ignore_index=True)
                    # Walls summary is included here because it's used to determine the distance to walls
                    # Echoes summary is included here because it's used to determine the distance to others
                    all_summary1 = pandas.DataFrame.from_records((echoes_summary, walls_summary, calls_summary, echoes_summary2, walls_summary2))
                    all_pulses2 = copy.copy(all_pulses1)
                    all_summary2 = copy.copy(all_summary1)

        if condition in ['default', 'cohesion']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_pulses1 = pandas.concat((echoes, walls, calls, echoes2, walls2), ignore_index=True)
                all_summary1 = pandas.DataFrame.from_records((echoes_summary, calls_summary, walls_summary, echoes_summary2, walls_summary2))
                all_pulses2 = copy.copy(all_pulses1)
                all_summary2 = copy.copy(all_summary1)

        if condition == 'no_others':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_pulses1 = pandas.concat((echoes, walls, walls2), ignore_index=True)
                all_summary1 = pandas.DataFrame.from_records((echoes_summary, walls_summary, walls_summary2))
                all_pulses2 = copy.copy(all_pulses1)
                all_summary2 = copy.copy(all_summary1)

        if condition.startswith('perched'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_pulses1 = pandas.concat((echoes, walls, calls, echoes2, walls2, perched), ignore_index=True)
                all_summary1 = pandas.DataFrame.from_records((echoes_summary, calls_summary, walls_summary, echoes_summary2, walls_summary2, perched_summary))
                all_pulses2 = copy.copy(all_pulses1)
                all_summary2 = copy.copy(all_summary1)

        if condition == 'differential':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_pulses1 = pandas.concat((walls, walls2), ignore_index=True)
                all_summary1 = pandas.DataFrame.from_records((walls_summary, walls_summary2))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                all_pulses2 = pandas.concat((walls, walls2, echoes, calls, echoes2), ignore_index=True)
                all_summary2 = pandas.DataFrame.from_records((walls_summary, walls_summary2, echoes_summary, calls_summary, echoes_summary2))

        all_pulses1['focus_bat'] = index
        all_summary1['focus_bat'] = index
        all_summary1 = all_summary1.set_index('pulse_type')

        all_pulses2['focus_bat'] = index
        all_summary2['focus_bat'] = index
        all_summary2 = all_summary2.set_index('pulse_type')

        return all_pulses1, all_summary1, all_pulses2, all_summary2

    def export_bat_data(self):
        all_bats = self.all_bats
        all_data = []
        for bat in all_bats:
            data = bat.logger.export()
            all_data.append(data)
        all_data = pandas.concat(all_data, ignore_index=True)
        all_data = all_data.sort_values(by=['focus_bat', 'simulation_step'])
        return all_data

    def simulation_step(self, bat_order=None, one_bat=False):
        if bat_order is None:
            bat_order = list(range(self.nr_of_bats))
            random.shuffle(bat_order)

        for bat_nr in bat_order:
            current_bat = self.__getitem__(bat_nr)
            pulses1, meta1, pulses2, meta2 = self.get_pulses_single_bat(bat_nr)
            delays = Tadarida.get_delays(pulses1, max_delay=self.pulse_interval)
            delays1 = delays * 1.0

            left_cochlear1, _ = Tadarida.pulses2cochlear(pulses1, side='left', condition=self.condition, delays=delays, max_delay=self.processing_interval)
            right_cochlear1, _ = Tadarida.pulses2cochlear(pulses1, side='right', condition=self.condition, delays=delays, max_delay=self.processing_interval)

            max_across_ears1 = numpy.maximum(left_cochlear1, right_cochlear1)
            max_index = numpy.argmax(max_across_ears1)

            left_integrated1 = numpy.sum(left_cochlear1[max_index - half_width:max_index + half_width])
            right_integrated1 = numpy.sum(right_cochlear1[max_index - half_width:max_index + half_width])

            # in case of not differential
            left_integrated2 = left_integrated1 * 1.0
            right_integrated2 = right_integrated1 * 1.0

            if self.condition == 'differential':
                # in this case:
                # 1: walls, walls2,
                # 2: walls, walls2 + echoes, calls, echoes2
                delays = Tadarida.get_delays(pulses2, max_delay=self.pulse_interval)
                left_cochlear2, _ = Tadarida.pulses2cochlear(pulses2, side='left', condition=self.condition, delays=delays, max_delay=self.processing_interval)
                right_cochlear2, _ = Tadarida.pulses2cochlear(pulses2, side='right', condition=self.condition, delays=delays, max_delay=self.processing_interval)

                max_across_ears2 = numpy.maximum(left_cochlear2, right_cochlear2)
                max_index = numpy.argmax(max_across_ears2)

                left_integrated2 = numpy.sum(left_cochlear2[max_index - half_width:max_index + half_width])
                right_integrated2 = numpy.sum(right_cochlear2[max_index - half_width:max_index + half_width])

            # subsample this for memory reasons
            # To save storage, pulses are only stored every 20 simulation steps.
            # At each stored step, the pulses from all bats are added to all_pulse_data.
            # This means the total number of stored pulses is greater than the number of simulation steps.
            if self.simulation_step_nr % 20 == 0: self.all_pulse_data.append(pulses1)

            maxed1 = max(left_integrated1, right_integrated1)
            maxed2 = max(left_integrated2, right_integrated2)

            diffed1 = right_integrated1 - left_integrated1
            diffed2 = right_integrated2 - left_integrated2

            # This is for all conditions except the differential condition and the insensitive condition
            sign = numpy.sign(diffed1)
            # This is for the differential condition
            if (maxed2 - maxed1) > 350: sign = numpy.sign(diffed2)  # 2: walls, walls2 + echoes, calls, echoes2
            # This is for the cohesion condition
            if self.condition == 'cohesion':
                sign = - numpy.sign(diffed1)
                if maxed1 > 700: sign = numpy.sign(diffed1)

            if self.condition == 'muted':
                focus_bat_calls = bat_nr % 2 == self.simulation_step_nr % 2
                if not focus_bat_calls: sign = -sign

            current_bat.speed = current_bat.speed + ((8 / 33) * 0.5)
            if current_bat.speed > self.max_speed: current_bat.speed = self.max_speed
            if current_bat.speed < 1: current_bat.speed = 1
            speed = current_bat.speed

            magnitude = Holderied.turning_rate(speed)
            if self.condition == 'random': sign = random.choice([-1, 1])
            speed_rot = magnitude * sign

            step = str(self.simulation_step_nr)
            replication = str(self.replication)
            condition = str(self.condition)
            bat = str(bat_nr)
            output_str = f'Bat {bat}\tStep {step} of replication {replication}, condition {condition}. {maxed1}, {maxed2}'
            print(output_str)

            current_bat.motion(body_y=speed_rot, speed=speed, time=self.pulse_interval, steps=1, update_log=False)
            corrected = False
            current_position = current_bat.position.copy()
            in_arena = self.in_arena(current_position[0:2])

            if not in_arena and self.arena_type != 'open':
                outside_xy = list(current_position[0:2])
                outside_xy_entry = [current_bat.id, self.simulation_step_nr] + outside_xy
                self.outside_positions.append(outside_xy_entry)
                corrected = True
                current_bat.speed = 1
                current_bat_z = current_bat.z

                # get mean bat position
                bat_positions = self.bat_positions
                mean_bat_position = numpy.mean(bat_positions, axis=0)
                mean_bat_position[2] = current_bat_z
                # get edge point
                edge_distance, edge_point = self.get_closest_point(bat_nr, type='edge')
                edge_point[2] = current_bat_z
                # get center point
                center_distance, center_point = self.get_closest_point(bat_nr, type='center')
                temp_frame = Geometry.Frame(position=edge_point)

                # set target and face it
                target = center_point
                az, el, dist = temp_frame.world2frame(target, spherical=True)
                current_bat.goto(x=edge_point[0], y=edge_point[1], yaw=az, update_log=False)

            closest_wall = meta1.loc['walls', 'min_distance']
            closest_echo = meta2.loc['echoes', 'min_distance']

            center_distance, _ = self.get_closest_point(bat_nr, type='center')

            # To get some properties of the pulses before turning them into cochlear signals
            if condition == 'differential':
                processed = process_pulses(pulses2, self.condition)
            else:
                processed = process_pulses(pulses1, self.condition)

            extra_log_items = copy.copy(processed)

            extra_log_items['corrected'] = corrected
            extra_log_items['left_integrated'] = left_integrated1
            extra_log_items['right_integrated'] = right_integrated1
            extra_log_items['max_integrated'] = maxed1
            extra_log_items['closest_echo'] = closest_echo
            extra_log_items['closest_wall'] = closest_wall
            extra_log_items['closest'] = min(closest_echo, closest_wall)
            extra_log_items['simulation_step'] = self.simulation_step_nr
            extra_log_items['speed'] = speed
            extra_log_items['maxed'] = max(maxed1, maxed2)
            extra_log_items['center_distance'] = center_distance
            current_bat.update_log(extra_log_entries=extra_log_items)
            if one_bat: break
        self.simulation_step_nr = self.simulation_step_nr + 1

        selected_delays = delays1 < self.processing_interval
        # select those rows from pulse1 where selected_delays is True
        pulses1 = pulses1[selected_delays]

        return pulses1  # return the pulses for the last bat (which will be random)


def process_pulses(pulses, condition):
    # selected = pulses.query('delay > 0.004')
    selected = pulses.query('maxed > -1100000')
    if condition == 'no_others': selected = selected.query('pulse_type != "echoes"')
    maxed = selected['maxed']
    max_id = maxed.idxmax()
    max_row = selected.loc[max_id, :]

    left_sum = 10 * numpy.log10(numpy.sum(10 ** (selected.left.values / 10)))
    right_sum = 10 * numpy.log10(numpy.sum(10 ** (selected.right.values / 10)))

    pulse_type = max_row['pulse_type']
    pulse_type_code = None
    if pulse_type == 'echoes': pulse_type_code = 0
    if pulse_type == 'walls': pulse_type_code = 1
    if pulse_type == 'calls': pulse_type_code = 2
    if pulse_type == 'echoes2': pulse_type_code = 3
    if pulse_type == 'walls2': pulse_type_code = 4
    if pulse_type == 'perched': pulse_type_code = 5

    results = {}
    results['pulse_type'] = pulse_type
    results['pulse_type_code'] = pulse_type_code
    results['focus_bat'] = max_row['focus_bat']
    results['diffed'] = max_row['diffed']
    results['distance'] = max_row['distance']
    results['maxed'] = max_row['maxed']
    results['summed'] = max_row['summed']
    results['left_sum'] = left_sum
    results['right_sum'] = right_sum
    # results['row'] = max_row
    return results
