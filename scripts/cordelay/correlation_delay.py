import numpy as np
from scripts.trj_data_utils import get_traj


def calculate_directional_correlation_delay(trajectories, bird_i, bird_j, tau):
    if tau < 0:
        bird_i, bird_j = bird_j, bird_i
        tau = -tau
    # for traj in trajectories:
    #     print(traj[0][0])
    trajectory_i = get_traj(trajectories, bird_i)
    trajectory_j = get_traj(trajectories, bird_j)

    frames_i, velocities_i = np.array(trajectory_i)[:, 1], np.array(
        trajectory_i)[:, 5:8]  # Extract frames and velocities from trajectory_i
    frames_j, velocities_j = np.array(trajectory_j)[:, 1], np.array(
        trajectory_j)[:, 5:8]  # Extract frames and velocities from trajectory_j

    # Find the common frames between trajectory_i and trajectory_j
    common_frames = np.intersect1d(frames_i, frames_j, assume_unique=True)

    # Select velocities_i for common frames
    velocities_i_common = velocities_i[np.isin(frames_i, common_frames)]
    # Select velocities_j for common frames
    velocities_j_common = velocities_j[np.isin(frames_j, common_frames)]

    if tau > len(velocities_i_common):
        raise ValueError(
            'tau is larger than the number of points in the trajectory')

    # Shift velocities_j_common by tau frames
    velocities_j_delayed = velocities_j_common[tau:]

    # Compute the numerator
    numerator = np.sum(
        velocities_i_common[: -tau if tau != 0 else None] * velocities_j_delayed)
    denominator = np.linalg.norm(velocities_i_common[: -tau if tau != 0 else None]) * np.linalg.norm(
        velocities_j_delayed)  # Compute the denominator

    c_ij_tau = numerator / denominator  # Calculate Cij(τ)

    return c_ij_tau


def calculate_correlation_pairs_multiple_tau(trajectories, bird_i, bird_j, tau_values):
    correlation_pairs = {}  # Dictionary to store the correlation values for each pair
    c_ij_values = []  # List to store the c_ij_tau values

    for tau in tau_values:
        if len(get_traj(trajectories, bird_i)) < 60 or len(get_traj(trajectories, bird_j)) < 60:
            print('Skipped. Too few points in trajectories.')
            continue
        c_ij_tau = calculate_directional_correlation_delay(
            trajectories, bird_i, bird_j, tau)
        c_ij_values.append(c_ij_tau)

    pair_key = (bird_i, bird_j)
    correlation_pairs[pair_key] = c_ij_values
    return correlation_pairs


def calculate_heatmap_data(data, bird_ids, tau_values):
    num_birds = len(bird_ids)
    heatmap_data = np.zeros((num_birds, num_birds))
    tau_max_data = np.zeros((num_birds, num_birds))

    # Calculate the maximum Cij(tau) values and corresponding tau values for each pair of birds
    for i, bird_i in enumerate(bird_ids):
        for j, bird_j in enumerate(bird_ids[i + 1:], start=i + 1):
            correlation_pairs = calculate_correlation_pairs_multiple_tau(
                data, bird_i, bird_j, tau_values)
            c_ij_values = correlation_pairs[(bird_i, bird_j)]
            max_c_ij_tau = max(c_ij_values)
            max_tau_index = c_ij_values.index(max_c_ij_tau)
            max_tau = tau_values[max_tau_index]
            heatmap_data[i, j] = max_c_ij_tau
            heatmap_data[j, i] = max_c_ij_tau  # Fill in the symmetric entry
            tau_max_data[i, j] = max_tau
            tau_max_data[j, i] = max_tau  # Fill in the symmetric entry

    return heatmap_data, tau_max_data


def calculate_normalized_velocity(velocity):
    norm = np.linalg.norm(velocity)
    if norm == 0:
        return velocity
    return velocity / norm
