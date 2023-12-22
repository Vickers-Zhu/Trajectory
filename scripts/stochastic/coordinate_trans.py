import numpy as np


def transform_point(a, b, c, x, y, z):
    # Step 1: Translation
    translated = np.array([a, b, c]) - np.array([x, y, z])

    # Step 2: Calculate new Z-axis (normalized)
    z_prime = np.array([0, y, z])
    z_prime = z_prime / np.linalg.norm(z_prime)

    # Calculate new Y-axis (cross product of new Z-axis and old X-axis)
    x_axis = np.array([1, 0, 0])
    y_prime = np.cross(z_prime, x_axis)
    y_prime = y_prime / np.linalg.norm(y_prime)

    # Rotation matrix (using new Z-axis and Y-axis, and old X-axis)
    rotation_matrix = np.column_stack((x_axis, y_prime, z_prime))

    # Step 3: Apply Rotation
    rotated = rotation_matrix @ translated

    return rotated


def rotate_around_x(point, phi):
    """
    Rotate a point around the X-axis by an angle phi (in radians).

    Args:
    point (tuple): The point to rotate, represented as (x, y, z).
    phi (float): The rotation angle in radians.

    Returns:
    np.array: The rotated point.
    """
    # Define the rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    # Convert point to numpy array for matrix multiplication
    point_array = np.array(point)

    # Apply the rotation
    rotated_point = rotation_matrix @ point_array

    return rotated_point


def rotate_x_gaussian_distribution(mean, covariance, phi):
    """
    Rotate a multivariate Gaussian distribution around the X-axis.

    Args:
    mean (np.array): Mean vector of the Gaussian distribution.
    covariance (np.array): Covariance matrix of the Gaussian distribution.
    phi (float): The rotation angle in radians.

    Returns:
    tuple: The rotated mean and covariance matrix.
    """
    # Define the rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    # Rotate the mean vector
    rotated_mean = rotation_matrix @ mean

    # Rotate the covariance matrix
    rotated_covariance = rotation_matrix @ covariance @ rotation_matrix.T

    return rotated_mean, rotated_covariance


# Example usage
a, b, c = 0, 1, 11  # Point in original coordinate system
x, y, z = 0, 1, 12  # New origin point
transformed_point = transform_point(a, b, c, x, y, z)

print("Transformed Point:", transformed_point)
