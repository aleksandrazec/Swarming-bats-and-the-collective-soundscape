import numpy
import math
import copy
from matplotlib import pyplot
from pyBat import Misc, Logger, Geometry, SimpleLogger

numpy.set_printoptions(precision=3)


class Bat:
    def __init__(self, id=None, logger='default'):
        self.id = id
        self.body_abs = Geometry.Frame()
        self.head_rel = Geometry.Frame()
        self.speed = 0
        self.logger = Logger.Logger()
        if logger == 'simple': self.logger = SimpleLogger.SimpleLogger()
        self.clock = 0

        # NOT IMPLEMENTED YET
        # self.auto_limit_angles = False
        # self.head_yaw_limit = 90
        # self.head_pitch_limit = 45
        # self.head_roll_limit = 60
        #
        # self.body_pitch_limit = 30
        # self.body_roll_limit = 30

    @property
    def logger_type(self):
        description = str(type(self.logger))
        if 'Simple' in description: return 'simple'
        return 'default'

    @property
    def head_abs(self):
        head_abs = Geometry.Frame(quaternion=self.body_abs.quaternion, position=self.body_abs.position)
        head_abs.apply_quaternion(self.head_rel.quaternion)
        return head_abs

    @property
    def x(self):
        return self.body_abs.x

    @property
    def y(self):
        return self.body_abs.y

    @property
    def z(self):
        return self.body_abs.z

    @property
    def position(self):
        return self.body_abs.position

    def translate(self, axis, distance):
        self.body_abs.translate(axis, distance)

    def motion(self, **kwargs):
        time = kwargs.pop('time', 0)
        nr_of_steps = math.ceil(time / 0.01)
        nr_of_steps = kwargs.pop('steps', nr_of_steps)
        speed = kwargs.pop('speed', 0)
        # Parameters are velocities
        body_yaw = kwargs.pop('body_y', 0)
        body_pitch = kwargs.pop('body_p', 0)
        body_roll = kwargs.pop('body_r', 0)

        head_yaw = kwargs.pop('head_y', 0)
        head_pitch = kwargs.pop('head_p', 0)
        head_roll = kwargs.pop('head_r', 0)

        update_log = kwargs.pop('update_log', True)

        extra_log_entries = kwargs.pop('log', {})

        keys = kwargs.keys()
        if len(keys) > 0: raise ValueError('Unused keyword arguments passed')

        # head_yaw = head_yaw - body_yaw
        # head_pitch = head_pitch - body_pitch
        # head_roll = head_roll - body_roll

        if nr_of_steps > 1:
            time_steps = numpy.linspace(0, time, nr_of_steps + 1)
            time_steps = numpy.diff(time_steps)
        else:
            time_steps = [time]

        for step in time_steps:
            self.body_abs.move(time=step, yaw=body_yaw, pitch=body_pitch, roll=body_roll, speed=speed)
            self.head_rel.move(time=step, yaw=head_yaw, pitch=head_pitch, roll=head_roll, speed=0)
            # if self.auto_limit_angles:
            #     self.head_rel.limit_rotations(yaw_limit=self.head_yaw_limit, pitch_limit=self.head_pitch_limit, roll_limit=self.head_roll_limit)
            #     self.body_abs.limit_rotations(pitch_limit=self.body_pitch_limit, roll_limit=self.body_roll_limit)

            self.clock += step
            if update_log: self.update_log(extra_log_entries)

    def goto(self, x=None, y=None, z=None, yaw=None, pitch=None, roll=None, update_log=True, extra_log_entries=False):
        # if self.auto_limit_angles:
        #     if pitch is not None and : print('Warning: requested pitch too large!')
        self.body_abs.goto(x, y, z, yaw, pitch, roll)
        if not extra_log_entries: extra_log_entries = {}
        if update_log: self.update_log(extra_log_entries)

    def bat2world(self, bat, frame='a'):
        if frame.startswith('b'): world = self.body_abs.frame2world(bat)
        if frame.startswith('seed_points'): world = self.head_rel.frame2world(bat)
        if frame.startswith('a'): world = self.head_abs.frame2world(bat)
        return world

    def world2bat(self, world, frame='a', spherical=False):
        if frame.startswith('b'): bat = self.body_abs.world2frame(world, spherical=spherical)
        if frame.startswith('seed_points'): bat = self.head_rel.world2frame(world, spherical=spherical)
        if frame.startswith('a'): bat = self.head_abs.world2frame(world, spherical=spherical)
        return bat

    def remove_self_from_list(self, bats):
        if not Misc.iterable(bats): bats = [bats]
        new_list = []
        for bat in bats:
            if bat.id != self.id: new_list.append(bat)
        return new_list

    def get_others_abs_positions(self, bats, frame='a'):
        new_list = self.remove_self_from_list(bats)
        positions = []
        ids = []
        for bat in new_list:
            positions.append(bat.position)
            ids.append(bat.id)
        positions = numpy.array(positions)
        return positions, ids

    def get_others_relative_positions(self, bats, frame='a', spherical=False, remove_self=False):
        bats = copy.copy(bats)
        if remove_self: bats = self.remove_self_from_list(bats)
        positions = []
        ids = []
        for bat in bats:
            relative = self.world2bat(bat.position, frame, spherical)
            positions.append(relative)
            ids.append(bat.id)
        positions = numpy.array(positions)
        return positions, ids

    def get_self_relative_position(self, bats, frame='a', spherical=False, remove_self=False):
        bats = copy.copy(bats)
        if remove_self: bats = self.remove_self_from_list(bats)
        positions = []
        ids = []
        for bat in bats:
            relative = bat.world2bat(self.position, frame, spherical)
            positions.append(relative)
            ids.append(bat.id)
        positions = numpy.array(positions)
        return positions, ids

    def get_closest_bats(self, bats, n=1, return_values='bat'):
        new_list = self.remove_self_from_list(bats)
        result, _ = self.get_others_relative_positions(new_list, frame='a', spherical=True)
        azimuths = result[:, 0]
        elevations = result[:, 1]
        distances = result[:, 2]
        sort_indices = distances.argsort()
        selected_indices = sort_indices[0:n]
        selected_indices = numpy.sort(selected_indices)
        if return_values == 'index':
            return selected_indices
        if return_values == 'bat':
            selected_bats = []
            for i in selected_indices: selected_bats.append(new_list[i])
            return selected_bats
        if return_values == 'loc':
            selected_azs = azimuths[selected_indices]
            selected_els = elevations[selected_indices]
            selected_dists = distances[selected_indices]
            if n == 1:
                selected_azs = float(selected_azs)
                selected_els = float(selected_els)
                selected_dists = float(selected_dists)
            return selected_azs, selected_els, selected_dists

    def update_log(self, extra_log_entries={}):
        body = self.body_abs.euler
        head = self.head_rel.euler
        total = self.head_abs.euler
        # Add time data
        self.logger['time'] = self.clock
        # Add position data
        self.logger['x', 'y', 'z'] = [self.x, self.y, self.z]
        # Add body data
        self.logger['rxb', 'ryb', 'rzb'] = body
        # Add head data
        self.logger['rxh', 'ryh', 'rzh'] = head
        # Add total data
        self.logger['rxt', 'ryt', 'rzt'] = total
        # Add body quaternion
        self.logger['b0', 'b1', 'b2', 'b3'] = self.body_abs.quaternion
        # Add head quaternion
        self.logger['r0', 'r1', 'r2', 'r3'] = self.head_rel.quaternion
        # Add total quaternion
        self.logger['h0', 'h1', 'h2', 'h3'] = self.head_abs.quaternion

        for key in extra_log_entries.keys(): self.logger[key] = extra_log_entries[key]

    def get_history(self, field, time_stamps=None, gradient=False, unwrap=True, absolute=False):
        apply_unwrap = False
        if self.logger_type == 'simple': return self.logger.get(field)
        angle_fields = ['a_b', 'e_b', 'a_h', 'e_h', 'a_t', 'e_t']
        if field in angle_fields and unwrap: apply_unwrap = True
        data = self.logger.get_history(field, time_stamps, gradient, apply_unwrap, False, absolute)
        return data

    def get_history_vectors(self, time_stamps=None, frame='b'):
        if frame.startswith('b'): labels = ['b0', 'b1', 'b2', 'b3']
        if frame.startswith('h'): labels = ['h0', 'h1', 'h2', 'h3']
        q0 = self.get_history(labels[0], time_stamps=time_stamps)
        q1 = self.get_history(labels[1], time_stamps=time_stamps)
        q2 = self.get_history(labels[2], time_stamps=time_stamps)
        q3 = self.get_history(labels[3], time_stamps=time_stamps)

        x = self.get_history('x', time_stamps=time_stamps)
        y = self.get_history('y', time_stamps=time_stamps)
        z = self.get_history('z', time_stamps=time_stamps)

        quaternions = numpy.row_stack((q0, q1, q2, q3))
        positions = numpy.row_stack((x, y, z))

        quaternions = numpy.transpose(quaternions)
        positions = numpy.transpose(positions)

        quaternions = numpy.asmatrix(quaternions)
        positions = numpy.asmatrix(positions)
        vectors = numpy.zeros(shape=positions.shape)
        for index in range(0, positions.shape[0]):
            quaternion = Misc.mat2array(quaternions[index, :])
            position = Misc.mat2array(positions[index, :])
            frame = Geometry.Frame(quaternion=quaternion, position=position)
            vector = frame.get_relative_direction_vector()
            vectors[index, :] = vector

        quaternions = Misc.mat2array(quaternions)
        positions = Misc.mat2array(positions)
        vectors = Misc.mat2array(vectors)
        return vectors, positions, quaternions

    def get_path(self):
        x = self.logger['x']
        y = self.logger['y']
        z = self.logger['z']
        return x, y, z

    def plot_path(self):
        x, y, z = self.get_path()
        path_figure = pyplot.figure()
        axis = path_figure.gca(projection='3d')
        axis.plot(x, y, z, marker='o')
        axis.set_xlabel('X Label')
        axis.set_ylabel('Y Label')
        axis.set_zlabel('Z Label')

    def plot_quiver(self, time_stamps=None, view='top', length=0.05, auto_axis=True):
        t = self.logger['time']
        mn = t.min()
        mx = t.max()
        if time_stamps is None: time_stamps = self.logger['time']
        if type(time_stamps) == float: time_stamps = numpy.arange(mn, mx, time_stamps)
        vectors_head_abs, start_head_abs, _ = self.get_history_vectors(time_stamps=time_stamps, frame='h')
        vectors_body_abs, start_body_abs, _ = self.get_history_vectors(time_stamps=time_stamps, frame='b')

        vectors_head_abs *= length
        vectors_body_abs *= length

        max_body = numpy.max(numpy.abs(start_body_abs))
        max_total = numpy.max(numpy.abs(start_head_abs))
        max_value = max(max_body, max_total) + 0.5

        number = vectors_head_abs.shape[0]

        if view.startswith('3'):
            fig = pyplot.figure()
            ax = fig.gca(projection='3d')

        for index in range(0, number):
            t_x0 = start_head_abs[index, 0]
            t_y0 = start_head_abs[index, 1]
            t_z0 = start_head_abs[index, 2]
            t_x1 = vectors_head_abs[index, 0]
            t_y1 = vectors_head_abs[index, 1]
            t_z1 = vectors_head_abs[index, 2]

            b_x0 = start_body_abs[index, 0]
            b_y0 = start_body_abs[index, 1]
            b_z0 = start_body_abs[index, 2]
            b_x1 = vectors_body_abs[index, 0]
            b_y1 = vectors_body_abs[index, 1]
            b_z1 = vectors_body_abs[index, 2]

            if view.startswith('t'):
                a = pyplot.arrow(b_x0, b_y0, b_x1, b_y1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
                b = pyplot.arrow(t_x0, t_y0, t_x1, t_y1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)
                pyplot.xlabel('x')
                pyplot.ylabel('y')

            if view.startswith('s'):
                a = pyplot.arrow(b_x0, b_z0, b_x1, b_z1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
                b = pyplot.arrow(t_x0, t_z0, t_x1, t_z1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)
                pyplot.xlabel('x')
                pyplot.ylabel('z')

            if view.startswith('3'):
                ax.quiver(b_x0, b_y0, b_z0, b_x1, b_y1, b_z1, color='k', alpha=0.75)
                ax.quiver(t_x0, t_y0, t_z0, t_x1, t_y1, t_z1, color='r', alpha=0.75)

        if view.startswith('3'):
            pyplot.xlabel('x')
            pyplot.ylabel('y')
            return

        if auto_axis:
            pyplot.xlim(-max_value, max_value)
            pyplot.ylim(-max_value, max_value)
            ax = pyplot.gca()
            ax.set_aspect(1)

        results = {}
        results['vectors_body'] = vectors_body_abs
        results['vectors_head'] = vectors_head_abs
        results['start_head'] = start_head_abs
        results['start_body'] = start_body_abs
        return results, a, b

# b = Bat()
# b.goto(1,1,0,yaw=0, pitch=30)
# b.motion(time=1.41, speed=1, steps=3)
# #b.motion(time=0.5, speed=1, steps=3)
# #b.plot_quiver('all', view='top')
# #pyplot.grid()
# #pyplot.show()
# print(b.x)


# a = numpy.array([0, 0, 1])
# b = Bat()
#
# pitches = []
# yaws = []
# rolls = []
# for x in range(0, 10):
#     b.motion(head_pitch=45, speed=1, time=0.1)
#     b.head_rel.limit_rotations(pitch_limit=33)
#     rolls.append(b.head_abs.euler[0])
#     pitches.append(b.head_abs.euler[1])
#     yaws.append(b.head_abs.euler[2])
#
#
# b.plot_quiver(view='s')
# #pyplot.figure()
# #b.plot_quiver(view='t')
# print(b.body_abs.spherical_angles)
# print(b.head_rel.spherical_angles)
# print(b.head_abs.spherical_angles)
#
# print(b.head_abs.euler)


# Rotate.test_frame()
