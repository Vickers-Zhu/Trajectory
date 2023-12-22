import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.use('Qt5Agg')


class StereoscopicSensorSystem:
    def __init__(self, pixel_size, focal_length, height, distance_between_lenses):
        self.P = pixel_size  # Pixel size in meters
        self.F = focal_length  # Focal Length in meters
        self.Z = height  # Height in meters
        self.D = distance_between_lenses  # Distance between lenses in meters

        # Refactored coordinates
        self.left_len = (-self.D/2, 0, 0)
        self.right_len = (self.D/2, 0, 0)
        self.left_sensor = (-self.D/2, 0, -self.F)
        self.right_sensor = (self.D/2, 0, -self.F)
        self.star = (0, 0, self.Z)

        # Calculate dz and dr
        self.dz = (self.Z**2) * (self.P/2) / (self.F * self.D)
        self.dr = self.Z * (self.P/2) / self.F
        # # Calculate the coordinates for ebot and etop
        self.ebot = (0, 0, -self.dz + self.Z)
        self.etop = (0, 0, self.dz + self.Z)

    def interpolate_point_on_line_through_star(self, start, star, z_value):
        direction = np.array(star) - np.array(start)
        t = (z_value - start[2]) / direction[2]
        x = start[0] + t * direction[0]
        y = start[1] + t * direction[1]
        return (x, y, z_value)

    def calculate_intersections(self):
        intersection_left = self.interpolate_point_on_line_through_star(
            self.left_len, self.star, self.left_sensor[2])
        intersection_right = self.interpolate_point_on_line_through_star(
            self.right_len, self.star, self.right_sensor[2])
        return intersection_left, intersection_right

    def plot_system(self):
        intersection_left, intersection_right = self.calculate_intersections()

        # Preparing data for plotting
        lenses_x = [self.left_len[0], self.right_len[0]]
        lenses_y = [self.left_len[1], self.right_len[1]]
        lenses_z = [self.left_len[2], self.right_len[2]]

        # sensors_x = [self.left_sensor[0], self.right_sensor[0]]
        # sensors_y = [self.left_sensor[1], self.right_sensor[1]]
        # sensors_z = [self.left_sensor[2], self.right_sensor[2]]

        # Prepare vectors for plotting
        vectors_x_left = [self.left_len[0], self.star[0], intersection_left[0]]
        vectors_y_left = [self.left_len[1], self.star[1], intersection_left[1]]
        vectors_z_left = [self.left_len[2], self.star[2], intersection_left[2]]
        vectors_x_right = [self.right_len[0],
                           self.star[0], intersection_right[0]]
        vectors_y_right = [self.right_len[1],
                           self.star[1], intersection_right[1]]
        vectors_z_right = [self.right_len[2],
                           self.star[2], intersection_right[2]]

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
        ax.plot([self.etop[0], self.left_len[0]], [self.etop[1], self.left_len[1]], [
                self.etop[2], self.left_len[2]], color='purple', linestyle='-', label='Etop to Left Lens')
        ax.plot([self.etop[0], self.right_len[0]], [self.etop[1], self.right_len[1]], [
                self.etop[2], self.right_len[2]], color='purple', linestyle='-', label='Etop to Right Lens')
        ax.plot([self.ebot[0], self.left_len[0]], [self.ebot[1], self.left_len[1]], [
                self.ebot[2], self.left_len[2]], color='cyan', linestyle='-', label='Ebot to Left Lens')
        ax.plot([self.ebot[0], self.right_len[0]], [self.ebot[1], self.right_len[1]], [
                self.ebot[2], self.right_len[2]], color='cyan', linestyle='-', label='Ebot to Right Lens')

        # Mark the intersection points on the sensors' plane
        ax.scatter(*intersection_left, color='green', s=100,
                   marker='X', label='Left Intersection')
        ax.scatter(*intersection_right, color='purple', s=100,
                   marker='X', label='Right Intersection')

        # Draw the circle around the star
        # Generating points on the circle in 3D
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.star[0] + self.dr * np.cos(theta)
        y = self.star[1] + self.dr * np.sin(theta)
        z = np.full_like(0, self.star[2])

        # Plotting the circle
        ax.plot(x, y, z, linestyle=':', color='red',
                label='Errors around Star')
        # Draw the star
        ax.scatter(self.star[0], self.star[1], self.star[2],
                   color='orange', s=1, label='Star')

        # Set the axes labels and limits
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])

        # Set the ticks for the lenses
        ax.set_xticks(lenses_x)
        ax.set_xticklabels(['Left Lens', 'Right Lens'])

        # Set the legend and title
        ax.legend(loc='upper left')
        ax.set_title('3D Stereoscopic Sensor System Representation')

        plt.show()


# # Create an instance of the class with given parameters
# system = StereoscopicSensorSystem(8.2e-6, 2.4e-2, 100, 2)

# # Plot the system
# system.plot_system()
