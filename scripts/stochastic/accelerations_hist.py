from scripts.trj_data_utils import get_traj, calculate_acceleration, calculate_acceleration_multi, find_trajectory_ids, get_trajs_by_ids, calculate_acceleration_fly_direction
from matplotlib import colormaps
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_acceleration(data, ID):
    selected_trajectory = get_traj(data, ID)
    acceleration_data = calculate_acceleration_fly_direction(
        selected_trajectory)
    colormap = colormaps.get_cmap('tab20')

    trj_id = acceleration_data[0][0]
    frames = [frame[1] for frame in acceleration_data]
    xaccs = [frame[2] for frame in acceleration_data]
    yaccs = [frame[3] for frame in acceleration_data]
    zaccs = [frame[4] for frame in acceleration_data]
    color = colormap(0)
    plt.plot(frames, xaccs, color=color,
             label=f"trj_id: {trj_id} - X Acceleration")
    plt.plot(frames, yaccs, color=color,
             linestyle=':', label=f"trj_id: {trj_id} - Y Acceleration")
    plt.plot(frames, zaccs, color=color,
             linestyle='--', label=f"trj_id: {trj_id} - Z Acceleration")

    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_all_accelerations_hist(data):
    valid_ids = find_trajectory_ids(data, 10)
    selected_trajectories = get_trajs_by_ids(data, valid_ids)
    acceleration_data = calculate_acceleration_multi(selected_trajectories)
    flat_acceleration_data = np.array([
        item for sublist in acceleration_data for item in sublist])
    # Create histograms for all three axes on a single graph
    plt.figure(figsize=(8, 15))
    print("Front dir acce: ", [item[3] for item in flat_acceleration_data])
    plt.hist([item[2] for item in flat_acceleration_data], bins=20, density=True,
             color='blue', alpha=0.5, label='X-axis')
    plt.hist([item[3] for item in flat_acceleration_data], bins=20, density=True,
             color='green', alpha=0.5, label='Y-axis')
    plt.hist([item[4] for item in flat_acceleration_data], bins=20, density=True,
             color='red', alpha=0.5, label='Z-axis')
    plot_gaussian([item[2] for item in flat_acceleration_data], 'blue')
    plot_gaussian([item[3] for item in flat_acceleration_data], 'green')
    plot_gaussian([item[4] for item in flat_acceleration_data], 'red')

    plt.title('Histograms of Accelerations in X, Y, and Z Axes')
    plt.xlabel('Acceleration')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the histogram
    plt.show()


def plot_accelerations_hist(data, id):
    '''
        Plots the histogram of accelerations for a single trajectory
        Parameters:    
            data: the data to be used
            id: the id of the trajectory to be plotted   

        Returns:       None 
    '''
    selected_trajectory = get_traj(data, id)
    acceleration_data = calculate_acceleration_fly_direction(
        selected_trajectory)
    flat_acceleration_data = np.array(acceleration_data)

    # Round the accelerations to a certain precision (e.g., 0.01)
    xaccelerations = [item[2] for item in flat_acceleration_data]
    yaccelerations = [item[3] for item in flat_acceleration_data]
    zaccelerations = [item[4] for item in flat_acceleration_data]

    # precision = 0.01
    # rounded_x_accelerations = np.round(
    #     np.array(xaccelerations) / precision) * precision
    # rounded_y_accelerations = np.round(
    #     np.array(yaccelerations) / precision) * precision
    # rounded_z_accelerations = np.round(
    #     np.array(zaccelerations) / precision) * precision
    # Create histograms for all three axes on a single graph
    plt.hist(xaccelerations, bins=30, density=True,
             color='blue', alpha=0.5, label='X-axis')
    plt.hist(yaccelerations, bins=30, density=True,
             color='green', alpha=0.5, label='Y-axis')
    plt.hist(zaccelerations, bins=30, density=True,
             color='red', alpha=0.5, label='Z-axis')

    plt.title('Histograms of Accelerations in X, Y, and Z Axes')
    plt.xlabel('Acceleration')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the histogram
    plt.show()


def hist_debug():

    # Generate random acceleration data for demonstration (replace this with your actual data)
    np.random.seed(0)
    accelerations = np.random.randn(1000)  # Example: 1000 random accelerations

    # Round the accelerations to a certain precision (e.g., 0.01)
    precision = 0.01
    rounded_accelerations = np.round(accelerations / precision) * precision

    # Create a histogram
    plt.hist(rounded_accelerations, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Accelerations (Rounded to 0.01 Precision)')
    plt.xlabel('Acceleration')
    plt.ylabel('Frequency')

    # Show the histogram
    plt.show()


def plot_gaussian(data, color):
    # Example data, replace with your actual dataset
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    # Plotting the mean
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    plt.plot(x, norm.pdf(x, mean, std_dev),
             label='Gaussian Distribution', color=color)
