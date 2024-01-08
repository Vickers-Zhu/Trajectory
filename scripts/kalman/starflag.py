import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
from ..stochastic.coordinate_trans import angle_with_z_axis, rotate_around_x, rotate_x_gaussian_distribution
from ..stochastic.gaussian import plot_3d_gaussian
# matplotlib.use('Qt5Agg')


class StereoscopicSensorSystem:
    def __init__(self, pixel_size, focal_length, distance_between_lenses, obj_pos, angle=None):
        self.P = pixel_size  # Pixel size in meters
        self.F = focal_length  # Focal Length in meters
        self.Z = np.sqrt(obj_pos[1]**2 + obj_pos[2]**2)  # Height from x axis
        self.D = distance_between_lenses  # Distance between lenses in meters

        # Refactored coordinates
        self.left_len = (-self.D/2, 0, 0)
        self.right_len = (self.D/2, 0, 0)
        self.left_sensor = (-self.D/2, 0, -self.F)
        self.right_sensor = (self.D/2, 0, -self.F)
        self.star = obj_pos

        # Calculate dz and dr
        self.dz = (self.Z**2) * (self.P/2) / (self.F * self.D)
        self.dr = self.Z * (self.P/2) / self.F
        # # Calculate the coordinates for errbot and errtop
        self.dir = -np.deg2rad(angle) if angle is not None else - \
            angle_with_z_axis(obj_pos[1], obj_pos[2])
        self.ebot = rotate_around_x(
            (obj_pos[0], 0, -self.dz + self.Z), -angle_with_z_axis(obj_pos[1], obj_pos[2]))
        self.etop = rotate_around_x(
            (obj_pos[0], 0, self.dz + self.Z), -angle_with_z_axis(obj_pos[1], obj_pos[2]))

    def interpolate_point_on_line_through_star(self, start, star, z_value):
        # Convert to numpy arrays for vectorized operations
        start = np.array(start)
        star = np.array(star)

        # Calculate the direction vector of the line
        direction = star - start

        # Avoid division by zero if the direction vector is parallel to the XY plane
        if direction[2] == 0:
            raise ValueError(
                "The line is parallel to the XY plane and does not intersect at z_value.")

        # Calculate the parameter t at the intersection point
        t = (z_value - start[2]) / direction[2]

        # Calculate the x and y coordinates of the intersection point
        x = start[0] + t * direction[0]
        y = start[1] + t * direction[1]

        return (x, y, z_value)

    def calculate_intersections_on_sensor(self):
        intersection_left = self.interpolate_point_on_line_through_star(
            self.left_len, self.star, self.left_sensor[2])
        intersection_right = self.interpolate_point_on_line_through_star(
            self.right_len, self.star, self.right_sensor[2])
        return intersection_left, intersection_right

    def plot_system(self):
        intersection_left, intersection_right = self.calculate_intersections_on_sensor()

        # Preparing data for plotting
        lenses_x = [self.left_len[0], self.right_len[0]]
        lenses_y = [self.left_len[1], self.right_len[1]]
        lenses_z = [self.left_len[2], self.right_len[2]]

        # sensors_x = [self.left_sensor[0], self.right_sensor[0]]
        # sensors_y = [self.left_sensor[1], self.right_sensor[1]]
        # sensors_z = [self.left_sensor[2], self.right_sensor[2]]

        # Prepare vectors for plotting
        vectors_x_left = [intersection_left[0], self.star[0]]
        vectors_y_left = [intersection_left[1], self.star[1]]
        vectors_z_left = [intersection_left[2], self.star[2]]
        vectors_x_right = [intersection_right[0],
                           self.star[0]]
        vectors_y_right = [intersection_right[1],
                           self.star[1]]
        vectors_z_right = [intersection_right[2],
                           self.star[2]]

        # Plotting
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw the lenses with a unique marker
        ax.scatter(lenses_x, lenses_y, lenses_z, color='black',
                   s=100, marker='^', label='Lenses')
        ax.plot(lenses_x, lenses_y, lenses_z, color='grey', linewidth=1)

        # Draw vectors from the lenses to the star and extend them to the sensors' plane
        ax.plot(vectors_x_left, vectors_y_left, vectors_z_left, color='blue',
                linestyle='--', linewidth=1, marker='o', label='Vector to Star from Left Lens')
        ax.plot(vectors_x_right, vectors_y_right, vectors_z_right, color='red',
                linestyle='--', linewidth=1, marker='o', label='Vector to Star from Right Lens')

        # # Draw lines from etop and ebot to the left and right lenses
        etop_intersections_left = self.interpolate_point_on_line_through_star(
            self.left_len, self.etop, self.left_sensor[2])
        etop_intersections_right = self.interpolate_point_on_line_through_star(
            self.right_len, self.etop, self.right_sensor[2])
        ebot_intersections_left = self.interpolate_point_on_line_through_star(
            self.left_len, self.ebot, self.left_sensor[2])
        ebot_intersections_right = self.interpolate_point_on_line_through_star(
            self.right_len, self.ebot, self.right_sensor[2])

        # Plot lines from etop to intersections
        ax.plot([self.etop[0], etop_intersections_left[0]], [self.etop[1], etop_intersections_left[1]], [
                self.etop[2], etop_intersections_left[2]], color='purple', linestyle='-', label='Etop to Left Intersection')
        ax.plot([self.etop[0], etop_intersections_right[0]], [self.etop[1], etop_intersections_right[1]], [
                self.etop[2], etop_intersections_right[2]], color='purple', linestyle='-', label='Etop to Right Intersection')

        # Plot lines from ebot to intersections
        ax.plot([self.ebot[0], ebot_intersections_left[0]], [self.ebot[1], ebot_intersections_left[1]], [
                self.ebot[2], ebot_intersections_left[2]], color='cyan', linestyle='-', label='Ebot to Left Intersection')
        ax.plot([self.ebot[0], ebot_intersections_right[0]], [self.ebot[1], ebot_intersections_right[1]], [
                self.ebot[2], ebot_intersections_right[2]], color='cyan', linestyle='-', label='Ebot to Right Intersection')

        # Mark the intersection points on the sensors' plane
        ax.scatter(*intersection_left, color='green', s=100,
                   marker='X', label='Left Intersection')
        ax.scatter(*intersection_right, color='purple', s=100,
                   marker='X', label='Right Intersection')

        self.plot_error_circle(ax)

        # Draw the star
        ax.scatter(self.star[0], self.star[1], self.star[2],
                   color='orange', s=1, label='Star')
        self.plot_errors(ax)
        # Set the axes labels and limits
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # Setting the aspect ratio to be equal for all axes
        ax.set_box_aspect([1, 1, 1])
        # Set the legend and title
        ax.legend(loc='upper left')
        ax.set_title('3D Stereoscopic Sensor System Representation')

        plt.show()

    def plot_error_circle(self, ax):
        # Draw the circle around the star
        # Generating points on the circle in 3D
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 0 + self.dr * np.cos(theta)
        y = 0 + self.dr * np.sin(theta)
        z = np.full_like(theta, self.Z)
        points = np.vstack((x, y, z)).T

        rotated_points = np.array(
            [rotate_around_x(point, self.dir) for point in points])

        # Plotting the circle
        ax.plot(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], linestyle=':', color='red',
                label='Errors around Star')

    def plot_errors(self, ax):
        mu = np.array([0, 0, 0])
        Sigma = np.array(
            [[(2*self.dr)**2, 0, 0], [0, (2*self.dr)**2, 0], [0, 0, (2*self.dz)**2]])
        mu, Sigma = rotate_x_gaussian_distribution(mu, Sigma, self.dir)
        plot_3d_gaussian(mu, Sigma, ax=ax, offset=np.array(
            [self.star[0], self.star[1], self.star[2]]))


# Example usage
obj_pos = (47.159596, 178.670423, 59.549856)
system = StereoscopicSensorSystem(3.45e-6, 1.2e-2, 2.085, obj_pos, 90-29.52)

# # Plot the system
system.plot_system()
