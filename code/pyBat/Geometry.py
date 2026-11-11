import math
import numpy
import copy
from transforms3d import euler
from transforms3d import quaternions
from pyBat import Misc
from matplotlib import pyplot

class Frame:
    def __init__(self, *, quaternion=None, position=None):
        self.quaternion = copy.copy(quaternion)
        self.position = copy.copy(position)
        if quaternion is None: self.quaternion = numpy.array([1, 0, 0, 0])
        if position is None: self.position = numpy.array([0, 0, 0])
        self.motion_vector = numpy.array([1, 0, 0])

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @property
    def yaw(self):
        z_rotation = self.euler[2]
        return z_rotation

    @property
    def pitch(self):
        y_rotation = self.euler[1]
        return y_rotation

    @property
    def roll(self):
        x_rotation = self.euler[0]
        return x_rotation

    @property
    def rotation_matrix_frame2world(self):
        mat = quaternions.quat2mat(self.quaternion)
        mat = numpy.squeeze(mat)
        return mat

    @property
    def rotation_matrix_world2frame(self):
        conjugate = quaternions.qconjugate(self.quaternion)
        return quaternions.quat2mat(conjugate)

    @property
    def direction_vector(self):
        # DO NOT NORMALIZE OR SCALE THE OUTPUT VECTOR!!!
        # IF YOU NEED A SCALED VERSION (GIVEN LENGTH USE get_direction_vector())
        vector = numpy.array([1, 0, 0])
        vector = self.frame2world(vector)
        return vector

    def get_direction_vector(self, length=1):
        vector = numpy.array([length, 0, 0])
        vector = self.frame2world(vector)
        return vector

    def get_relative_direction_vector(self, length=1):
        vector = numpy.array([length, 0, 0])
        vector = self.frame2world(vector)
        vector = vector - self.position
        return vector

    # @property
    # DONT USE THIS FUNCTION - elevation and pitch are not measured in the same way.
    # def spherical_angles(self):
    #     vector = self.direction_vector
    #     azimuth, elevation, _ = cart2sph(vector[0], vector[1], vector[2])
    #     return azimuth, elevation

    @property
    def euler(self):
        angles = euler.quat2euler(self.quaternion, axes='sxyz')
        angles = numpy.array(angles)
        angles = numpy.rad2deg(angles)
        return angles

    def frame2world(self, frame):
        frame = copy.copy(frame)
        frame = frame.astype(float)
        frame = numpy.asmatrix(frame)
        frame = frame.transpose()
        world = numpy.dot(self.rotation_matrix_frame2world, frame)
        world = world.transpose()
        world[:, 0] = world[:, 0] + self.x
        world[:, 1] = world[:, 1] + self.y
        world[:, 2] = world[:, 2] + self.z
        world = Misc.mat2array(world)
        return world

    def world2frame(self, world, spherical=False):
        world = copy.copy(world)
        world = world.astype(float)
        world = numpy.asmatrix(world)
        world[:, 0] = world[:, 0] - self.x
        world[:, 1] = world[:, 1] - self.y
        world[:, 2] = world[:, 2] - self.z
        world = world.transpose()
        frame = numpy.dot(self.rotation_matrix_world2frame, world)
        frame = frame.transpose()
        if spherical:
            az, el, dist = mat2sph(frame)
            result = numpy.column_stack((az, el, dist))
            result = numpy.squeeze(result)
            return result
        frame = Misc.mat2array(frame)
        return frame

    def eccentricity(self, azimuth, elevation):
        x, y, z = sph2cart(azimuth, elevation, 1)
        vector = numpy.array([x, y, z])
        angle = Misc.angle_between(vector, self.motion_vector)
        return angle

    def move(self, time=0, yaw=0, pitch=0, roll=0, speed=0):
        rotation = make_quaternion(yaw * time, pitch * time, roll * time)
        self.quaternion = quaternions.qmult(self.quaternion, rotation)
        self.motion_vector = Misc.normalize_vector(self.motion_vector) #in case the motion vector was manually supplied
        body_step = numpy.dot(self.rotation_matrix_frame2world, self.motion_vector) * speed * time
        self.position = self.position + body_step

    def translate(self, axis='x', distance=0):
        if axis == 'x': motion_vector = numpy.array([1, 0, 0])
        if axis == 'y': motion_vector = numpy.array([0, 1, 0])
        if axis == 'z': motion_vector = numpy.array([0, 0, 1])
        motion_vector = Misc.normalize_vector(motion_vector)
        body_step = numpy.dot(self.rotation_matrix_frame2world, motion_vector) * distance
        self.position = self.position + body_step

    def limit_rotations(self, yaw_limit=180, pitch_limit=90, roll_limit=180):
        current_euler_angles = self.euler
        x_rotation = current_euler_angles[0]
        y_rotation = current_euler_angles[1]
        z_rotation = current_euler_angles[2]
        if abs(x_rotation) > roll_limit: x_rotation = roll_limit * Misc.sign(x_rotation)
        if abs(y_rotation) > pitch_limit: y_rotation = pitch_limit * Misc.sign(y_rotation)
        if abs(z_rotation) > yaw_limit: z_rotation = yaw_limit * Misc.sign(z_rotation)
        x_rotation = numpy.deg2rad(x_rotation)
        y_rotation = numpy.deg2rad(y_rotation)
        z_rotation = numpy.deg2rad(z_rotation)
        quaternion = euler.euler2quat(x_rotation, y_rotation, z_rotation, axes='sxyz')
        self.quaternion = copy.copy(quaternion)

    def apply_quaternion(self, quaternion):
        quaternion = copy.copy(quaternion)
        self.quaternion = quaternions.qmult(self.quaternion, quaternion)

    def apply_translation(self, position):
        position = copy.copy(position)
        self.position = self.position + position

    def set_rotations(self, yaw=0, pitch=0, roll=0):
        rotation = make_quaternion(yaw, pitch, roll)
        self.quaternion = numpy.array([1, 0, 0, 0])
        self.motion_vector = numpy.array([1, 0, 0]) # I am not sure why I do this here.
        self.apply_quaternion(rotation)

    def goto(self, x=None, y=None, z=None, yaw=None, pitch=None, roll=None):
        if x is None: x = self.x
        if y is None: y = self.y
        if z is None: z = self.z
        if yaw is None: yaw = self.yaw
        if pitch is None: pitch = self.pitch
        if roll is None: roll = self.roll
        self.position = numpy.array([x, y, z])
        self.set_rotations(yaw, pitch, roll)

    def plot_xy(self, length=1):
        vector = numpy.array([1, 0, 0]) * length
        x = self.x
        y = self.y
        pointer = self.frame2world(vector)
        pyplot.scatter(x, y)
        pyplot.plot([x, pointer[0]], [y, pointer[1]])


def make_quaternion(yaw=0, pitch=0, roll=0):
    pitch_rad = numpy.deg2rad(pitch)
    yaw_rad = numpy.deg2rad(yaw)
    roll_rad = numpy.deg2rad(roll)
    q = euler.euler2quat(roll_rad, pitch_rad, yaw_rad, axes='sxyz')
    return q

# This was the old version
# def make_quaternion(yaw=0, pitch=0, roll=0):
#     pitch_rad = numpy.deg2rad(pitch)
#     yaw_rad = numpy.deg2rad(yaw)
#     roll_rad = numpy.deg2rad(roll)
#     rotation_vector = numpy.array([roll_rad, pitch_rad, yaw_rad])
#     theta = math.sqrt(numpy.dot(rotation_vector, rotation_vector))
#     # Deal with small thetas
#     if theta > 0: rotation_vector = rotation_vector / theta
#     if theta < 1e-6: rotation_vector = numpy.array([1, 0, 0])
#     rotation = quaternions.axangle2quat(rotation_vector, theta, True)
#     return rotation


def rotate_points_cart(x, y, z, yaw=0, pitch=0, roll=0):
    points = numpy.column_stack((x, y, z))
    quaternion = make_quaternion(-yaw, -pitch, -roll)
    matrix = quaternions.quat2mat(quaternion)
    points = points @ matrix
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return x, y, z


def rotate_points_sph(az, el, dist=None, yaw=0, pitch=0, roll=0):
    if dist is None: dist = numpy.ones(az.shape)
    x, y, z = sph2cart(az, el, dist)
    x, y, z = rotate_points_cart(x, y, z, yaw, pitch, roll)
    azimuth, elevation, distance = cart2sph(x, y, z)
    return azimuth, elevation, distance


def map2pi(angles):
    rads = numpy.deg2rad(angles)
    phases = (rads + numpy.pi) % (2 * numpy.pi) - numpy.pi
    angles = numpy.rad2deg(phases)
    return angles


def sph2cart(azimuth, elevation, distance):
    azimuth = numpy.deg2rad(azimuth)
    elevation = numpy.deg2rad(elevation)
    x = distance * numpy.cos(elevation) * numpy.cos(azimuth)
    y = distance * numpy.cos(elevation) * numpy.sin(azimuth)
    z = distance * numpy.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    x2 = numpy.square(x)
    y2 = numpy.square(y)
    z2 = numpy.square(z)
    azimuth = numpy.arctan2(y, x)
    elevation = numpy.arctan2(z, numpy.sqrt(x2 + y2))
    distance = numpy.sqrt(x2 + y2 + z2)
    azimuth = numpy.rad2deg(azimuth)
    elevation = numpy.rad2deg(elevation)
    return azimuth, elevation, distance


def mat2sph(matrix):
    x = matrix[:, 0]
    y = matrix[:, 1]
    z = matrix[:, 2]
    azimuth, elevation, distance = cart2sph(x, y, z)
    azimuth = Misc.mat2array(azimuth)
    elevation = Misc.mat2array(elevation)
    distance = Misc.mat2array(distance)
    return azimuth, elevation, distance

# These are functions that relate to the Moss paper on prey persuit
def phi_range(alpha):
    alpha = numpy.array([alpha])
    alpha = numpy.deg2rad(alpha)
    alpha = numpy.mod(alpha,2*numpy.pi)
    idx = alpha>numpy.pi
    alpha[idx] = alpha[idx]-2*numpy.pi
    alpha = numpy.rad2deg(alpha)
    return alpha[0]

def phi2_range(alpha):
    alpha = numpy.array([alpha])
    alpha = numpy.deg2rad(alpha)
    positive_input = (alpha > 0)
    alpha = numpy.mod(alpha, 2*numpy.pi)
    alpha[alpha == 0 * positive_input] = 2*numpy.pi
    alpha = numpy.rad2deg(alpha)
    return alpha[0]




# def test_frame():
#     reference = numpy.array(([1, 2, 3], [3, 4, 5]))
#     pitch = 90
#     yaw = 90
#     roll = 90
#     steps = 200
#
#     frame = Frame()
#     for step in range(0, steps):
#
#         p1 = random.random()
#         p2 = random.random()
#         p3 = random.random()
#
#         if p1 < 0.1: yaw = yaw * -1
#         if p2 < 0.1: pitch = pitch * -1
#         if p3 < 0.1: pitch = roll * -1
#
#         frame.move(time=1, speed=10, pitch=pitch, yaw=yaw, roll=roll)
#
#         frame_coordinates = frame.world2frame(reference)
#         world_coordinates = frame.frame2world(frame_coordinates)
#
#         print(frame_coordinates)
#         print(world_coordinates)
#         print('-----------------')
#
#         # m1 = frame.rotation_matrix_frame2world
#         # m2 = frame.rotation_matrix_world2frame
#         # call_result = numpy.dot(m1, m2)
#         # print(call_result)
#
#
# def test_cart2sph():
#     x = numpy.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
#     y = numpy.array([[1, 1, -1, -1], [1, 1, -1, -1]])
#     z = numpy.array([[1, -1, 1, -1], [1, -1, 1, -1]])
#     az, el, dist = cart2sph(x, y, z)
#     x2, y2, z2 = sph2cart(az, el, dist)
#     az, el, dist = cart2sph(x, y, z)
#     print(az)
#     print('---')
#     print(el)
#     print('---')
#     print(dist)

# a = numpy.array([[1, 0, 0],[0,0,1]])
# b = Frame()
# x = b.world2frame(a,True)
# print(x)


# # Quaternion implementation
# pitch_rad = numpy.deg2rad(pitch)
# yaw_rad = numpy.deg2rad(yaw)
# roll_rad = numpy.deg2rad(roll)
# rotation_vector = time * numpy.array([roll_rad, pitch_rad, yaw_rad])
# theta = math.sqrt(numpy.dot(rotation_vector, rotation_vector))
# # Deal with small thetas
# if theta > 0: rotation_vector = rotation_vector / theta
# if theta < 1e-6: rotation_vector = numpy.array([1, 0, 0])
# # New Quaternion
# rotation = quaternions.axangle2quat(rotation_vector, theta, True)
