import itertools
import math
import argparse
import os

import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Polygon
import scipy.ndimage

c = 2.3263  # -norminv(0.01)

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def in_circle(pt, circle_centre, r):
    return dist(pt, circle_centre) <= r

def count_pixels_in_circle(pixels, circle_centre, r):
    count = 0
    for px in pixels:
        if dist(px, circle_centre) <= r:
            count += 1
    return count

def dirs(neighbour_order):
    dirs_list = list(itertools.product(range(-neighbour_order, neighbour_order + 1), range(-neighbour_order, neighbour_order + 1)))
    dirs_list.remove((0, 0))
    return dirs_list

def angle(pt, circle_centre):
    return (math.atan2(pt[1] - circle_centre[1], pt[0] - circle_centre[0])*180/math.pi) % 360

def line_angle(line):
    vec = (line[0][0] - line[1][0], line[0][1] - line[1][1])
    return math.acos(vec[0]/dist(vec, (0, 0)))

def centroid(pts):
    centroid_x = float(sum([pt[0] for pt in pts]))/len(pts)
    centroid_y = float(sum([pt[1] for pt in pts]))/len(pts)
    return (centroid_x, centroid_y)

def arc_through_points(p1, p2, circle_centre, centroid):
    '''
    Determine which direction to draw arc: from point 1 to point 2,
    or from point 2 to point 1. This is done by imposing that the angle of the
    midpoint between the two points is between the two angles.

    In some circumstances the centre of the arc may be outside the polygon
    formed by the vertices, so the angles for determining the order are
    measured with respect to the centroid of the vertices instead.
    '''
    theta1 = angle(p1, circle_centre)
    theta2 = angle(p2, circle_centre)
    disp_vec = (p2[0] - p1[0], p2[1] - p1[1])
    midpoint = (p1[0] + disp_vec[0]/2, p1[1] + disp_vec[1]/2)
    theta_mid = angle(midpoint, centroid)
    theta1_centroid = angle(p1, centroid)
    theta2_centroid = angle(p2, centroid)
    if (theta1_centroid <= theta_mid <= theta2_centroid) or (theta_mid <= theta2_centroid <= theta1_centroid) or (theta2_centroid <= theta1_centroid <= theta_mid):
        return (theta1, theta2)
    else:
        return (theta2, theta1)

def polygonize(pts):
    '''
    Form a polygon from the given points by connecting them in order of angle
    from the centroid, and return the order of points as well as the polygon's
    perimeter.

    This should be equivalent to the convex hull when the set of points is
    convex, and is probably the shortest path connecting the points in many
    cases (i.e., the solution to the traveling salesman problem).
    '''
    centre = centroid(pts)
    polygon = sorted(pts, key=lambda pt: angle(pt, centre))
    return polygon

def circles_through_points(p1, p2, r):
    '''Based on http://rosettacode.org/wiki/Circles_of_given_radius_through_two_points#Python'''
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1

    q = math.sqrt(dx**2 + dy**2)

    # halfway point
    x3, y3 = (x1 + x2)/2, (y1 + y2)/2

    # distance along the mirror line
    d = math.sqrt(r**2 - (q/2)**2)

    c1 = (x3 - d*dy/q, y3 + d*dx/q)
    c2 = (x3 + d*dy/q, y3 - d*dx/q)
    return c1, c2

def get_circle(p1, p2, pixels):
    '''Get the smallest circle which contains the maximum number of the given
       pixels and passes through p1 and p2'''
    step_size = len(pixels)/50.
    area = len(pixels)/2.
    max_contained_pixels = 0
    while True:
        radius = math.sqrt(area/math.pi)
        if dist(p1, p2) > 2*radius:
            # The distance between the two points can't be larger than the
            # diameter of the desired circle
            area = math.pi*(dist(p1, p2)/2.)**2
            radius = dist(p1, p2)/2.
        circle1, circle2 = circles_through_points(p1, p2, radius)
        circle1_pixels = count_pixels_in_circle(pixels, circle1, radius)
        circle2_pixels = count_pixels_in_circle(pixels, circle2, radius)
        contained_pixels = max([circle1_pixels, circle2_pixels])
        if contained_pixels > max_contained_pixels:
            max_contained_pixels = contained_pixels
            area += step_size
        else:
            return (circle1 if circle1_pixels > circle2_pixels else circle2, radius)

def shift_matrix(matrix, shift_dir):
    if shift_dir == 'l':
        matrix = numpy.hstack([matrix[:, 1:], numpy.zeros([matrix.shape[0], 1])])
    elif shift_dir == 'r':
        matrix = numpy.hstack([numpy.zeros([matrix.shape[0], 1]), matrix[:, :-1]])
    elif shift_dir == 'u':
        matrix = numpy.vstack([matrix[1:, :], numpy.zeros([1, matrix.shape[1]])])
    elif shift_dir == 'd':
        matrix = numpy.vstack([numpy.zeros([1, matrix.shape[1]]), matrix[:-1, :]])
    return matrix

class Lattice:
    def __init__(self, data_file, size):
        self.data_dict = dict()
        self.width = size[0]
        self.height = size[1]
        self.types = []
        self.colours = []
        self.matrix = numpy.zeros(size, dtype=int)
        self.data = open(data_file).read().split('\n')[1:-1]
        for pix in self.data:
            fields = pix.split()
            cell = int(fields[0])
            x = int(fields[3])
            y = int(fields[5])
            cell_type = fields[2]
            self.matrix[x][y] = cell
            if cell not in self.data_dict.keys():
                self.data_dict[cell] = {'pixels': [], 'vertices': set(),
                                        'boundary_vertices': []}
                if cell_type in self.types:
                    self.data_dict[cell]['type'] = self.types.index(cell_type)
                else:
                    self.types.append(cell_type)
                    self.data_dict[cell]['type'] = self.types.index(cell_type)
            self.data_dict[cell]['pixels'].append((x, y))

        for cell_type in self.types:
            self.colours.append(numpy.random.rand(3, 1))

        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift(self, pt, shift_dir):
        shifted = (pt[0] + shift_dir[0], pt[1] + shift_dir[1])
        try:
            self.matrix[shifted]
            return shifted
        except:
            return None

    def adjacent_cells(self, pixel, neighbour_order=1):
        adj_cells = []

        for shift_dir in dirs(neighbour_order):
            shifted = self.shift(pixel, shift_dir)
            if shifted != None:
                adj_cell = self.matrix[shifted]
                if adj_cell not in adj_cells:
                    adj_cells.append(adj_cell)

        return list(set(adj_cells))

    def get_vertices(self):
        neighbour_count = numpy.zeros((self.height, self.width), dtype=int)

        for cell in [0] + list(self.data_dict.keys()):
            # The following matrix records how many neighbouring lattice sites
            # belong to the cell `cell`, for each lattice site
            interfaces = ((shift_matrix(self.matrix, 'u') == cell).astype(int) +
                          (shift_matrix(self.matrix, 'd') == cell).astype(int) +
                          (shift_matrix(self.matrix, 'l') == cell).astype(int) +
                          (shift_matrix(self.matrix, 'r') == cell).astype(int))

            # If the cell is the same one as that which a lattice site belongs
            # to, do not record it as a distinct neighbour
            interfaces -= 4*(self.matrix == cell).astype(int)

            # If any of the neighbouring lattice sites belongs to the cell,
            # add 1 to the neighbour count of that lattice site
            neighbour_count += (interfaces > 0).astype(int)

        vertex_regions, num_vertices = scipy.ndimage.label((neighbour_count == 2).astype(int))

        vertices = scipy.ndimage.measurements.center_of_mass(neighbour_count, vertex_regions, range(num_vertices + 1))

        for vertex in vertices:
            cell = self.matrix[vertex]
            adj_cells = self.adjacent_cells(vertex, 1)

            boundary = 0 in adj_cells
            if boundary:
                adj_cells.remove(0)
            for adj_cell in adj_cells:
                self.data_dict[adj_cell]['vertices'].add(vertex)
                if boundary:
                    self.data_dict[adj_cell]['boundary_vertices'].append(vertex)

    def polygonize_cells(self, arcs=False):
        lines = []
        for cell in self.data_dict.keys():
            vertices = self.data_dict[cell]['vertices']
            boundary_vertices = self.data_dict[cell]['boundary_vertices']
            polygon = polygonize(vertices)
            self.data_dict[cell]['polygon'] = polygon
            lines.extend(zip(polygon[:-1], polygon[1:]))
            lines.append((polygon[-1], polygon[0]))
            if len(boundary_vertices) == 2:
                try:
                    lines.remove((boundary_vertices[0], boundary_vertices[1]))
                except:
                    pass
                try:
                    lines.remove((boundary_vertices[1], boundary_vertices[0]))
                except:
                    pass
                if arcs:
                    pixels = self.data_dict[cell]['pixels']
                    circle, radius = get_circle(boundary_vertices[0], boundary_vertices[1], pixels)

                    # Determine which direction to draw the arc in
                    theta1, theta2 = arc_through_points(boundary_vertices[0], boundary_vertices[1], circle, centroid(vertices))

                    perimeter = sum([dist(p1, p2) for p1, p2 in zip(polygon[:-1], polygon[1:])]) + dist(polygon[0], polygon[-1]) - dist(boundary_vertices[0], boundary_vertices[1])
                    perimeter += 2*math.pi*radius*(((theta2 - theta1) % 360)/360.)
                    self.data_dict[cell]['perimeter'] = perimeter

                    lines.append((theta1, theta2, radius, circle))
            else:
                perimeter = sum([dist(p1, p2) for p1, p2 in zip(polygon[:-1], polygon[1:])]) + dist(polygon[0], polygon[-1])
                self.data_dict[cell]['perimeter'] = perimeter
        self.lines = set(list(map(tuple, map(sorted, lines))))
        return lines

    def plot(self):
        fig = plt.gcf()
        for line in self.lines:
            if len(line) == 2:
                plt.plot(list(zip(*line))[0], list(zip(*line))[1], color='black')

            elif len(line) == 4:
                theta1, theta2 = line[0], line[1]
                circle_centre = line[3]
                diameter = 2*line[2]

                arc = Arc(circle_centre, width=diameter, height=diameter,
                          theta1=theta1, theta2=theta2, color='black', linewidth=1)
                fig.gca().add_artist(arc)

        for cell_dict in self.data_dict.values():
            polygon = Polygon(cell_dict['polygon'],
                              color=self.colours[cell_dict['type']])
            fig.gca().add_artist(polygon)

        return fig

    def angle_distribution(self):
        '''
        See C. Arthur Williams Jr., "On the Choice of the Number and Width of
        Classes for the Chi-Square Test of Goodness of Fit", 1950 for the
        formula for number of bins used.
        '''
        angles = []
        for line in self.lines:
            if len(line) == 2:
                angles.append(line_angle(line))
        N = len(angles)
        k = 4*math.exp(0.2*math.log(2*(N**2)/c**2))
        hist, bins = numpy.histogram(angles, bins=k)
        chisq = sum([(freq - N/k)**2/(N/k) for freq in hist])
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        fig = plt.gcf()
        plt.bar(center, hist, align='center', width=width)
        plt.xlabel("Angle")
        plt.ylabel("Frequency")
        return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store', dest='file')
    parser.add_argument('--output', action='store', dest='output')
    parser.add_argument('--size', action='store', dest='size')
    results = parser.parse_args()
    size = tuple(map(int, results.size.split(',')))
    lattice = Lattice(results.file, size)
    lattice.get_vertices()
    lattice.polygonize_cells()
    name = os.path.splitext(os.path.basename(results.file))[0]
    lattice.plot().savefig(os.path.join(results.output, name + '.png'))
    plt.clf()
    lattice.angle_distribution().savefig(os.path.join(results.output, name + '_angles.png'))
