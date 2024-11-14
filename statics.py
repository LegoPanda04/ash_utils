import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrowPatch, Circle

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
        """
        This is not the same as the truss point and WILL NOT WORK WITH IT.
        :param x: x
        :param y: y
        :param z: z
        """
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


class TrussPoint:
    def __init__(self, coords, constraints, forces):
        """
        Point class to store all the different points in the truss.
        Format is TrussPoint(coords=[x,y], constraints=[x,y], forces=[x,y])
        :param coords: coordinates of the point: [x,y]
        :param constraints: constraints on the point as 1-0 or True-False: [x,y]
        :param forces: forces at the point: [x,y]
        """
        self.coords = np.array(coords)
        self.constraints = np.array(constraints)
        self.forces = np.array(forces)

        # Class variable to store the resultant force if needed.
        self.resultants = np.array([0, 0])

        # Class variable that keeps track of which rows in the solving matrix belong to it.
        self.row_index = None

    def plot_horizontal(self, ax, maxload, magnitude, color):
        """
        Internal horizontal arrow function.
        You should not need to mess with this.
        :param ax: ax from the plot().
        :param maxload: scaling value from the plot().
        :param magnitude: what variable should be used for plotting?
        :param color: color of the arrow.
        """
        offset_x = np.sign(magnitude[0]) * maxload
        ax.add_patch(FancyArrowPatch((self.coords[0] - offset_x, self.coords[1]), (self.coords[0], self.coords[1]),
                                     arrowstyle='->', mutation_scale=20, color=color, zorder=10))
        ax.text(self.coords[0] - offset_x, self.coords[1], str(round(abs(magnitude[0]), 2)), ha='center', va='center',
                zorder=10)

    def plot_vertical(self, ax, maxload, magnitude, color):
        """
        Internal vertical arrow function.
        You should not need to mess with this.
        :param ax: ax from the plot().
        :param maxload: scaling value from the plot().
        :param magnitude: what variable should be used for plotting?
        :param color: color of the arrow.
        """
        offset_y = np.sign(magnitude[1]) * maxload
        ax.add_patch(FancyArrowPatch((self.coords[0], self.coords[1] - offset_y), (self.coords[0], self.coords[1]),
                                     arrowstyle='->', mutation_scale=20, color=color, zorder=10))
        ax.text(self.coords[0], self.coords[1] - offset_y, str(round(abs(magnitude[1]), 2)), ha='center', va='center',
                zorder=10)

    def plot(self, ax, aspect, is_solved):
        """
        Plots the forces on a point
        :param ax: ax from the plot().
        :param aspect: scaling value from the plot().
        :param is_solved: boolean of whether to do the solved or unsolved version.
        """
        # Calculate maxload based on aspect
        maxload = aspect / 10

        # Fixed support
        if self.constraints[0] == 1 and self.constraints[1] == 1:
            if is_solved:
                self.plot_horizontal(ax, maxload, self.resultants, 'blue')
                self.plot_vertical(ax, maxload, self.resultants, 'blue')
            else:
                side_length = aspect / 20
                h = side_length * np.sqrt(3) / 2
                x_center = self.coords[0]
                y_center = self.coords[1] - 2 * h / 3
                ax.add_patch(
                    RegularPolygon((x_center, y_center), numVertices=3, radius=side_length / np.sqrt(3), orientation=0,
                                   color='b'))

        # Horizontal roller
        elif self.constraints[0] == 1 and self.constraints[1] == 0:
            if is_solved:
                self.plot_horizontal(ax, maxload, self.resultants, 'blue')
            else:
                radius = aspect / 40
                ax.add_patch(Circle((self.coords[0] + radius, self.coords[1]), radius=radius, color='b'))

        # Vertical roller
        elif self.constraints[0] == 0 and self.constraints[1] == 1:
            if is_solved:
                self.plot_vertical(ax, maxload, self.resultants, 'blue')
            else:
                radius = aspect / 40
                ax.add_patch(Circle((self.coords[0], self.coords[1] - radius), radius=radius, color='b'))

        # Horizontal force load
        if abs(self.forces[0]) > 0:
            self.plot_horizontal(ax, maxload, self.forces, 'red')

        # Vertical force load
        if abs(self.forces[1]) > 0:
            self.plot_vertical(ax, maxload, self.forces, 'red')


class TrussMember:
    def __init__(self, from_point, to_point):
        """
        Member class to store relationships between points.
        If your labels are coming out upside down, try flipping the order you input a and b.
        :param from_point: point a.
        :param to_point: point b.
        """
        self.from_point = from_point
        self.to_point = to_point
        self.length = np.linalg.norm(self.to_point.coords - self.from_point.coords)
        self.components = (self.to_point.coords - self.from_point.coords) / self.length

        # Class variable to store the tension in the member once calculated.
        self.tension = None

    def plot(self, ax, colors, is_solved):
        """
        Plots the truss member and internal forces if applicable.
        :param ax: ax from the plot().
        :param colors: color of the truss member.
        :param is_solved: boolean of whether to do the solved or unsolved version.
        """
        # Makes the x and y lists
        x_values = [self.from_point.coords[0], self.to_point.coords[0]]
        y_values = [self.from_point.coords[1], self.to_point.coords[1]]

        # Plots the line
        ax.plot(x_values, y_values, color=colors, linewidth=2)

        if is_solved:
            # Computes the midpoint and rotation angle.
            midpoint = (self.from_point.coords + self.to_point.coords) / 2
            deltas = self.to_point.coords - self.from_point.coords
            angle = np.rad2deg(np.arctan2(deltas[1], deltas[0]))

            # Plots text if needed
            ax.text(midpoint[0], midpoint[1], str(round(self.tension, 2)), rotation=angle, ha='center', va='center')


def extract_points(points):
    """
    Internal function to get the coordinates of the points out of the point class.
    You should not need to mess with this.
    :param points: list of point classes.
    :return: list of coordinates of the points.
    """
    return np.array([[point.coords[0], point.coords[1]] for point in points])


class Truss:
    def __init__(self, members):
        """
        Class containing all the truss members and is where the magic happens.
        Format is Truss([AB, BC, AC, ...]).
        There is currently not duplication checking.
        You do not need to pass in the points as those are contained inside the member classes.
        :param members: members of the truss as a list.
        """
        # Adds members as a self variable
        self.members = members

        # Creates an empty list of points
        self.points = []

        # Going through the list of members, add one copy of the point to the list, and update the point class to have an index pointing to the x and y respective indexes.
        # This is what enables the ability to pass in just the members and not the points.
        for mem in self.members:
            if not mem.from_point in self.points:
                self.points.append(mem.from_point)
                mem.from_point.row_index = 2 * self.points.index(mem.from_point)
            if not mem.to_point in self.points:
                self.points.append(mem.to_point)
                mem.to_point.row_index = 2 * self.points.index(mem.to_point)

    def solve(self, print_final=False):
        """
        Function to solve the truss.
        I may add options to display intermediate results at a later date.
        There is no output by default as each member and point has it's values updated instead.
        If you need the output, set print_final=True.
        :param print_final: prints the final output if needed, default is False.
        """
        # This is entirely because I do not want to redo all the instances of points and members to have self in front of them.
        points = self.points
        members = self.members

        # Calculating the amount of points and members we are dealing with.
        num_points = len(points)
        num_members = len(members)

        # Creates an array A twice as big as the points.
        # Creates an array b to hold the equals forces.
        A = np.zeros((num_points * 2, num_points * 2))
        b = np.zeros((num_points * 2, 1))

        # Adds the components of the internal forces to the array A.
        # This only operates on the left half, with each column being a member and each row being an x and y force at a point.
        # The second entry is mirrored because the forces are opposite.
        for col_index, mem in enumerate(members):
            A[mem.from_point.row_index:mem.from_point.row_index + 2, col_index] = mem.components
            A[mem.to_point.row_index:mem.to_point.row_index + 2, col_index] = -mem.components

        # Starts the counter at the right half of A, after the members,
        counter = num_members
        for point in points:
            # If there is a constraint than add a 1 to A.
            if point.constraints[0] > 0:
                A[point.row_index, counter] = point.constraints[0]
                counter += 1
            if point.constraints[1] > 0:
                A[point.row_index + 1, counter] = point.constraints[1]
                counter += 1

            # Populates the b array with the forces stored in the point class.
            b[point.row_index] = -point.forces[0]
            b[point.row_index + 1] = -point.forces[1]

        # Solves the equation
        solved = np.linalg.solve(A, b)

        # Populates the members to have the tension
        for row_index, mem in enumerate(members):
            mem.tension = solved[row_index, 0]

        # Resets the counter from earlier.
        counter = num_members
        for point in points:
            # The same logic from before but now we are storing the value of said forced.
            if point.constraints[0] > 0:
                point.resultants[0] = solved[counter, 0]
                counter += 1
            if point.constraints[1] > 0:
                point.resultants[1] = solved[counter, 0]
                counter += 1

        # IF YOU WANT TO SEE A AND B, UNCOMMENT BELOW!
        # print(A)
        # print(b)

        if print_final:
            # If you need it to return the matrix, change this to return solved
            print(solved)

    def plot(self, is_solved=False):
        """
        Plots the truss system.
        This function DOES NOT call self.solve().
        Make sure to call self.solve() if is_solved is True.
        :param is_solved: boolean of whether to do the solved or unsolved version, default is False.
        """
        # This is entirely because I do not want to redo all the instances of points and members to have self in front of them.
        points = self.points
        members = self.members

        # Sets colors for the members
        if is_solved:
            colors = [0.75, 0.75, 0.75]
        else:
            colors = [0.1, 0.1, 0.1]

        # Sets the aspect scaling.
        aspect = max(np.max(extract_points(points)[:, 0]) - np.min(extract_points(points)[:, 0]),
                     np.max(extract_points(points)[:, 1]) - np.min(extract_points(points)[:, 1]))

        # To be completely honest, I am not sure why we need to do plt.subplots(), but the patches won't work otherwise.
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.grid()

        # Plots the truss members.
        for member in members:
            member.plot(ax, colors, is_solved)

        # Plots the forces at each point.
        for point in points:
            point.plot(ax, aspect, is_solved)

        # Does some final formatting and plots.
        plt.axis('equal')
        plt.show()    