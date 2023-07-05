# import TrjSample
import matplotlib.pyplot as plt
import numpy as np
from correlation_delay import calculate_correlation_pairs_multiple_tau, get_traj


def read_trj_data(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    cnt = 0
    trj_prev = -1
    for line in lines:
        cols = line.split()
        if len(cols) > 1:  # skip empty line
            trj_id = int(cols[0])
            frame = int(cols[1])
            xpos = float(cols[6])  # depth
            ypos = float(cols[7])
            zpos = float(cols[8])  # height
            xvel = float(cols[12])
            yvel = float(cols[13])
            zvel = float(cols[14])
            n = int(cols[15])  # length of sequence

            if trj_prev != trj_id:
                if trj_prev != -1:  # very first one
                    data.append(trajectory)
                trajectory = []
                trj_prev = trj_id
                cnt += 1
            else:
                trajectory.append(
                    [trj_id, frame, xpos, ypos, zpos, xvel, yvel, zvel])
    return data


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


def create_tau_relations_matrix(heatmap_data, tau_max_data, bird_ids):
    tau_relations = np.zeros((len(bird_ids), len(bird_ids)))

    for i, bird_i in enumerate(bird_ids):
        for j, bird_j in enumerate(bird_ids[i + 1:], start=i + 1):
            if tau_max_data[i, j] < 0:
                tau_relations[j, i] = heatmap_data[i, j]
            else:
                tau_relations[i, j] = heatmap_data[i, j]

    return tau_relations


def find_related_most(tau_relations, bird_ids):
    related_most = {}

    for i, bird_i in enumerate(bird_ids):
        max_index = np.argmax(tau_relations[i, :])
        related_bird_id = bird_ids[max_index]
        related_most[bird_i] = related_bird_id

    return related_most


def find_trajectory_ids(trajectories, length_threshold):
    selected_ids = []
    for n in range(120):
        try:
            traj = get_traj(trajectories, n)
            if len(traj) > length_threshold:
                selected_ids.append(n)
        except ValueError:
            pass
    return selected_ids


def calculate_acceleration(selected_trajectories):
    acceleration_data = []

    for trajectory in selected_trajectories:
        acceleration_each = []
        for i in range(len(trajectory) - 1):
            trj_id = trajectory[i][0]
            frame = trajectory[i][1]
            xvel = trajectory[i][5]
            yvel = trajectory[i][6]
            zvel = trajectory[i][7]
            next_xvel = trajectory[i+1][5]
            next_yvel = trajectory[i+1][6]
            next_zvel = trajectory[i+1][7]
            xacc = next_xvel - xvel
            yacc = next_yvel - yvel
            zacc = next_zvel - zvel

            acceleration_each.append([trj_id, frame, xacc, yacc, zacc])
        acceleration_data.append(acceleration_each)

    return acceleration_data
