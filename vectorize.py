import itertools
import math

import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def in_circle(pt, circle_centre, r):
    return dist(pt, circle_centre) <= r

def dirs(neighbour_order):
    dirs_list = list(itertools.product(range(-neighbour_order, neighbour_order + 1), range(-neighbour_order, neighbour_order + 1)))
    dirs_list.remove((0, 0))
    return dirs_list

def angle(pt, circle_centre):
    return (math.atan2(pt[1] - circle_centre[1], pt[0] - circle_centre[0])*180/math.pi) % 360

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
    '''Naive traveling salesman. Improve if necessary.'''
    min_dist = float('inf')
    for path in itertools.permutations(pts):
        path_dist = sum([dist(pt1, pt2) for pt1, pt2 in zip(path[:-1], path[1:])]) + dist(path[0], path[-1])
        if path_dist < min_dist:
            min_dist = path_dist
            min_path = path
    return (min_path, min_dist)

def polygonize2(pts):
    centre = centroid(pts)
    return (sorted(pts, key=lambda pt: angle(pt, centre)), 0)

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
            area += step_size
            continue
        circle1, circle2 = circles_through_points(p1, p2, radius)
        circle1_pixels = list(map(lambda px: in_circle(px, circle1, radius), pixels)).count(True)
        circle2_pixels = list(map(lambda px: in_circle(px, circle2, radius), pixels)).count(True)
        contained_pixels = max([circle1_pixels, circle2_pixels])
        if contained_pixels > max_contained_pixels:
            max_contained_pixels = contained_pixels
            area += step_size
        else:
            return (circle1 if circle1_pixels > circle2_pixels else circle2, radius)

class Lattice:
    def __init__(self, data_file, size):
        self.data_dict = dict()
        self.width = size[0]
        self.height = size[1]
        self.matrix = numpy.zeros(size, dtype=int)
        self.data = open(data_file).read().split('\n')[1:-1]
        for pix in self.data:
            fields = pix.split()
            cell = int(fields[0])
            x = int(fields[3])
            y = int(fields[5])
            self.matrix[x][y] = cell
            if cell not in self.data_dict.keys():
                self.data_dict[cell] = {'pixels': [], 'vertices': set()}
            self.data_dict[cell]['pixels'].append((x, y))

        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift(self, pt, shift_dir):
        shifted = (pt[0] + shift_dir[0], pt[1] + shift_dir[1])
        try:
            self.matrix[shifted]
            return shifted
        except:
            return pt

    def shift_func_gen(self, shift_dir):
        def func(pt):
            shifted = (pt[0] + shift_dir[0], pt[1] + shift_dir[1])
            try:
                self.matrix[shifted]
                return shifted
            except:
                return pt
        return func

    def adjacent_cells(self, pixel, cell):
        adj_cells = []

        for shift_dir in self.dirs:
            adj_cell = self.matrix[self.shift(pixel, shift_dir)]
            if (adj_cell not in adj_cells) and (adj_cell != cell):
                adj_cells.append(adj_cell)

        return adj_cells

    def get_vertices(self, neighbour_order=1):
        vertices = []

        # First pass
        for cell in self.data_dict.keys():
            for pixel in self.data_dict[cell]['pixels']:
                adj_pixels = [self.shift(pixel, shift_dir) for shift_dir in dirs(neighbour_order)]
                adj_vertices = [shifted_pixel in vertices for shifted_pixel in adj_pixels]
                if any(adj_vertices):
                    self.data_dict[cell]['vertices'].add(adj_pixels[adj_vertices.index(True)])
                else:
                    adj_cells = self.adjacent_cells(pixel, cell)
                    if len(adj_cells) >= 2:
                        vertices.append(pixel)
                        self.data_dict[cell]['vertices'].add(pixel)

        # Second pass
        for cell in self.data_dict.keys():
            for pixel in self.data_dict[cell]['pixels']:
                if pixel not in vertices:
                    adj_pixels = [self.shift(pixel, shift_dir) for shift_dir in dirs(neighbour_order)]
                    adj_vertices = [shifted_pixel in vertices for shifted_pixel in adj_pixels]
                    if any(adj_vertices):
                        self.data_dict[cell]['vertices'].add(adj_pixels[adj_vertices.index(True)])

    def polygonize_cells(self):
        lines = []
        for cell in self.data_dict.keys():
            vertices = self.data_dict[cell]['vertices']
            boundary_vertices = list(filter(lambda vertex: 0 in self.adjacent_cells(vertex, cell), vertices))
            polygon = polygonize2(vertices)[0]
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
                pixels = self.data_dict[cell]['pixels']
                circle, radius = get_circle(boundary_vertices[0], boundary_vertices[1], pixels)
                lines.append((boundary_vertices[0], boundary_vertices[1], radius, circle, centroid(vertices)))
        return lines

    def plot(self):
        lines = set(self.polygonize_cells())
        fig = plt.gcf()
        for line in lines:
            if len(line) == 2:
                plt.plot(list(zip(*line))[0], list(zip(*line))[1], color='black')
            elif len(line) == 5:
                boundary_pt1, boundary_pt2 = line[0], line[1]
                circle_centre = line[3]
                diameter = 2*line[2]

                # Determine which direction to draw the arc in
                theta1, theta2 = arc_through_points(boundary_pt1, boundary_pt2, circle_centre, line[4])
                arc = Arc(circle_centre, width=diameter, height=diameter,
                          theta1=theta1, theta2=theta2, color='black', linewidth=1)
                fig.gca().add_artist(arc)
        return fig
