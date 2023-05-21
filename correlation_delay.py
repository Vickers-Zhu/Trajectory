import numpy as np

def calculate_directional_correlation_delay(trajectories, bird_i, bird_j, tau):
    trajectory_i = trajectories[bird_i]
    trajectory_j = trajectories[bird_j]
    
    frames_i, velocities_i = np.array(trajectory_i)[:, 0], np.array(trajectory_i)[:, 4:7]  # Extract frames and velocities from trajectory_i
    frames_j, velocities_j = np.array(trajectory_j)[:, 0], np.array(trajectory_j)[:, 4:7]  # Extract frames and velocities from trajectory_j
    
    # Find the common frames between trajectory_i and trajectory_j
    common_frames = np.intersect1d(frames_i, frames_j, assume_unique=True)
    
    velocities_i_common = velocities_i[np.isin(frames_i, common_frames)]  # Select velocities_i for common frames
    velocities_j_common = velocities_j[np.isin(frames_j, common_frames)]  # Select velocities_j for common frames
    
    velocities_j_delayed = velocities_j_common[tau:]  # Shift velocities_j_common by tau frames
    
    numerator = np.sum(velocities_i_common[:len(velocities_i_common)-tau] * velocities_j_delayed[:len(velocities_i_common)-tau])
    denominator = np.sqrt(np.mean(velocities_i_common[:-tau]) ** 2) * np.sqrt(np.mean(velocities_j_delayed) ** 2)  # Compute the denominator
    
    c_ij_tau = numerator / denominator  # Calculate Cij(τ)
    
    return c_ij_tau

def calculate_correlation_pairs_multiple_tau(trajectories, bird_i, bird_j, tau_values):
    correlation_pairs = {}  # Dictionary to store the correlation values for each pair

    for tau in tau_values:
        if len(trajectories[bird_i]) < 60 or len(trajectories[bird_j]) < 60:
            continue
        c_ij_tau = calculate_directional_correlation_delay(trajectories, bird_i, bird_j, tau)
        pair_key = (bird_i, bird_j)
        correlation_pairs[pair_key] = c_ij_tau

    return correlation_pairs

def calculate_correlation_pairs_tau(trajectories, tau):
    num_birds = len(trajectories)
    correlation_pairs = {}  # Dictionary to store the correlation values for each pair

    for bird_i in range(num_birds):
        if len(trajectories[bird_i]) < 60:  # Skip if the number of trajectories is less than 60
            continue
        for bird_j in range(bird_i + 1, num_birds):
            if len(trajectories[bird_j]) < 60:  # Skip if the number of trajectories is less than 60
                continue
            c_ij_tau = calculate_directional_correlation_delay(trajectories, bird_i, bird_j, tau)
            pair_key = (bird_i, bird_j)
            correlation_pairs[pair_key] = c_ij_tau

    return correlation_pairs
