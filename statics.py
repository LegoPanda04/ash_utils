import numpy as np
import matplotlib.pyplot as plt

class Vector:
    def __init__(self, comps, magnitude, unit_vector, angle_cosines, coords = np.zeros(3)):
        self.comps = comps
        self.magnitude = magnitude
        self.unit_vector = unit_vector
        self.angle_cosines = angle_cosines
        self.coords = coords

    @classmethod
    def input_components(cls, i=0, j=0 ,k=0):
        """
        Input function designed to take in components i, j, k.
        Initializes other vector components.
        :param i: component along x
        :param j: component along y
        :param k: component along z
        :return: calculated vector object
        """
        cls.comps = np.array([i, j, k])
        cls.magnitude = np.linalg.norm(cls.comps)
        cls.unit_vector = cls.comps / cls.magnitude
        cls.angle_cosines = np.arccos(cls.comps / cls.magnitude)
        
        return Vector(cls.comps, cls.magnitude, cls.unit_vector, cls.angle_cosines)
    
    @classmethod
    def input_nparray(cls, components):
        """
        Internal function to input components as a numpy array
        :param components: components as numpy array
        :return: calculated vector object
        """
        cls.comps = components
        cls.magnitude = np.linalg.norm(cls.comps)
        cls.unit_vector = cls.comps / cls.magnitude
        cls.angle_cosines = np.arccos(cls.comps / cls.magnitude)
        
        return Vector(cls.comps, cls.magnitude, cls.unit_vector, cls.angle_cosines)
    
    @classmethod
    def input_magnitude_angles(cls, magnitude, theta_y, theta_h):
        """
        Variant of input function designed to take in a magnitude and two angles.
        Initializes other vector components.
        :param magnitude: magnitude value
        :param theta_y: angle in DEGREES from +y to +x
        :param theta_h: angle in DEGREES from +x to +z
        :return: calculated vector object
        """
        theta_y = np.deg2rad(theta_y)
        theta_h = np.deg2rad(theta_h)

        cls.magnitude = magnitude
        cls.comps = cls.magnitude * np.array([np.sin(theta_y)*np.cos(theta_h), np.cos(theta_y), np.sin(theta_y)*np.sin(theta_h)])
        cls.unit_vector = cls.comps / cls.magnitude
        cls.angle_cosines = np.arccos(cls.comps / cls.magnitude)
        
        return Vector(cls.comps, cls.magnitude, cls.unit_vector, cls.angle_cosines)

    @classmethod
    def input_magnitude_cosines(cls, magnitude, theta_x, theta_y, theta_z):
        """
        Variant of input function designed to take in a magnitude and three angles.
        Initializes other vector components.
        :param magnitude: magnitude value
        :param theta_x: angle from +x in DEGREES
        :param theta_y: angle from +y in DEGREES
        :param theta_z: angle from +z in DEGREES
        :return: calculated vector object
        """
        cls.magnitude = magnitude
        cls.angle_cosines = np.array([np.deg2rad(theta_x), np.deg2rad(theta_y), np.deg2rad(theta_z)])
        cls.comps = cls.magnitude * np.cos(cls.angle_cosines)
        cls.unit_vector = cls.comps / cls.magnitude
        
        return Vector(cls.comps, cls.magnitude, cls.unit_vector, cls.angle_cosines)

    @property
    def dict(self):
        dict_comps = {'x': self.comps[0], 'y': self.comps[1], 'z': self.comps[2]}
        dict_unit = {'x': self.unit_vector[0], 'y': self.unit_vector[1], 'z': self.unit_vector[2]}
        dict_angle = {'x': self.angle_cosines[0], 'y': self.angle_cosines[1], 'z': self.angle_cosines[2]}
        return {'comp': dict_comps, 'unit': dict_unit, 'angle': dict_angle}
    
    def print_info(self, sig_figs=2):
        print(f'Info for Vector: {self}')
        print(f'Components: {np.round(self.comps, sig_figs)}')
        print(f'Magnitude: {np.round(self.magnitude, sig_figs)}')
        print(f'Angle: {np.round(np.rad2deg(self.angle_cosines), sig_figs)}')
        print(f'Unit: {np.round(self.unit_vector, sig_figs)}')
        print(f' ')

    def __add__(self, other):
        return self.input_nparray(self.comps + other.comps)
    
    def __sub__(self, other):
        return self.input_nparray(self.comps - other.comps)
    
    def __mul__(self, scalar):
        return self.input_nparray(self.comps * scalar)
    
    def __truediv__(self, scalar):
        return self.input_nparray(self.comps / scalar)
    
    def __pow__(self, scalar):
        return self.input_nparray(self.comps ** scalar)
    
    def __neg__(self):
        return self.input_nparray(self.comps * -1)
    
    def dot_product(self, other):
        return np.dot(self.comps, other.comps)
    
    def cross_product(self, other, print_intermediate=False):
        if print_intermediate:
            p = self.comps
            q = other.comps
            print(f'cross_product: {np.round(p, 4)} X {np.round(q, 4)}')
            print(f'cross_product: intermediate 1: ({p[1]:.4f}*{q[2]:.4f} - {p[2]:.4f}*{q[1]:.4f})i + ({p[2]:.4f}*{q[0]:.4f} - {p[0]:.4f}*{q[2]:.4f})j + ({p[0]:.4f}*{q[1]:.4f} - {p[1]:.4f}*{q[0]:.4f})k')
            print(f'cross_product: intermediate 2: ({p[1] * q[2]:.4f} - {p[2] * q[1]:.4f})i + ({p[2] * q[0]:.4f} - {p[0] * q[2]:.4f})j + ({p[0] * q[1]:.4f} - {p[1] * q[0]:.4f})k')
        return self.input_nparray(np.cross(self.comps, other.comps))
    
    def moment_origin(self, x, y, z):
        """
        Moment about a point, assuming the vector class is for force.
        Also assumes about origin
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :return: vector class of the moment.
        """
        return self.input_nparray(np.cross(np.array([x, y, z]), self.comps))

    def plot_vector(self, col='blue'):
        """
        Plots a projection of the vector in the xy plane.
        :param col: color, default 'blue'
        :return: pyplot arrow
        """
        plt.arrow(self.coords[0], self.coords[1], self.comps[0], self.comps[1], head_width=0.1, head_length=0.1, color=col, length_includes_head=True)

    def plot_coords_vector(self, x=0, y=0, z=0, col='blue'):
        """
        Identical to self.plot_vector, but it allows for updating coords
        :param x: x coordinate, default 0
        :param y: y coordinate, default 0
        :param z: z coordinate, default 0
        :param col: color, default 'blue'
        :return: pyplot arrow
        """
        self.coords = np.array([x, y, z])
        self.plot_vector(col)

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector.input_components(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector.input_components(self.x - other.x, self.y - other.y, self.z - other.z)

def line(a,b):
    """
    Makes a vector going from point a to b, which is b - a
    :param a: first point
    :param b: second point
    :return: vector from a to b
    """
    return b - a
