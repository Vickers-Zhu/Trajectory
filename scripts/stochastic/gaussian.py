import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from ..stochastic.coordinate_trans import rotate_x_gaussian_distribution


def generate_gaussian_samples(mu, Sigma, num_samples=10000):
    """
    Generate random samples for a 3D Gaussian distribution.

    Args:
    mu (np.array): Mean vector of the Gaussian distribution.
    Sigma (np.array): Covariance matrix of the Gaussian distribution.
    num_samples (int): Number of random samples to generate.

    Returns:
    np.array: Random samples generated from the Gaussian distribution.
    """
    # Standard deviation for each axis
    std_devs = np.sqrt(np.diag(Sigma))

    # Generating samples
    x = np.random.uniform(mu[0] - 2*std_devs[0],
                          mu[0] + 2*std_devs[0], num_samples)
    y = np.random.uniform(mu[1] - 2*std_devs[1],
                          mu[1] + 2*std_devs[1], num_samples)
    z = np.random.uniform(mu[2] - 2*std_devs[2],
                          mu[2] + 2*std_devs[2], num_samples)

    return np.vstack([x, y, z]).T

# Define the multivariate Gaussian function


def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian density value for a position.

    Args:
    pos (np.array): Position (or point) for which the density is computed.
    mu (np.array): Mean vector of the Gaussian distribution.
    Sigma (np.array): Covariance matrix of the Gaussian distribution.

    Returns:
    float: The density value at the specified position.
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


def plot_3d_gaussian(mu, Sigma, num_samples=10000, ax=None, offset=np.array([0, 0, 0])):
    """
    Plot a 3D Gaussian distribution.

    Args:
    mu (np.array): Mean vector of the Gaussian distribution.
    Sigma (np.array): Covariance matrix of the Gaussian distribution.
    num_samples (int): Number of random samples to generate for plotting.
    """

    # Calculate the density of these samples
    samples = generate_gaussian_samples(mu, Sigma, num_samples)

    # Extract x, y, and z coordinates
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    densities = np.array([multivariate_gaussian(sample, mu, Sigma)
                          for sample in samples])

    # Normalize densities for better visualization
    densities_normalized = 1 - densities / densities.max()

    # Scatter plot with size, color, and alpha based on density
    colors = plt.cm.plasma(densities_normalized)
    colors[:, 3] = 1 - densities_normalized

    if ax == None:
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Gaussian Distribution Visualization')

    # Draw the X, Y, and Z axes
    # axis_length = 5
    # X-axis
    # ax.plot([0, axis_length], [0, 0], [0, 0],
    #         color='red', lw=2, label='X-axis')
    # # Y-axis
    # ax.plot([0, 0], [0, axis_length], [0, 0],
    #         color='green', lw=2, label='Y-axis')
    # # Z-axis
    # ax.plot([0, 0], [0, 0], [0, axis_length],
    #         color='blue', lw=2, label='Z-axis')

    ax.scatter(x+offset[0], y+offset[1], z+offset[2], c=colors,
               marker='o', edgecolor='none', s=2)

    # plt.show()


def animate_rotation(mu, Sigma, num_samples=20000, frames=90):
    samples = generate_gaussian_samples(mu, Sigma, num_samples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Extract x, y, and z coordinates
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    scat = ax.scatter(x, y, z, c='blue', marker='o', edgecolor='none', s=2)

    def update(frame):
        # Rotate distribution
        rotated_mu, rotated_Sigma = rotate_x_gaussian_distribution(
            mu, Sigma, np.radians(-frame))

        # Update densities and colors
        densities = np.array([multivariate_gaussian(sample, rotated_mu, rotated_Sigma)
                              for sample in samples])
        densities_normalized = 1 - densities / densities.max()
        colors = plt.cm.plasma(densities_normalized)
        colors[:, 3] = 1 - densities_normalized

        # Update scatter plot
        scat.set_facecolor(colors)
        return scat,

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.linspace(
        0, -90, frames), blit=True)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Rotating 3D Gaussian Distribution')
    plt.show()
    # display(HTML(ani.to_html5_video()))


# # Example usage
# mu = np.array([6, 6, 6])
# Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# mu, Sigma = rotate_x_gaussian_distribution(mu, Sigma, np.radians(-90))
# plot_3d_gaussian(mu, Sigma)
# # animate_rotation(mu, Sigma)
