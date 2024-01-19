# import TrjSample
import matplotlib.pyplot as plt
import numpy as np


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
        print('index i: ', i, 'Max index, ', max_index)
        print(tau_relations[i, :])
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


def get_traj(trajectories, bird_id):
    for traj in trajectories:
        if len(traj) > 0 and len(traj[0]) > 0 and traj[0][0] == bird_id:
            return traj
    raise ValueError(f"No trajectory found with trj_id = {bird_id}")


def get_trajs_by_ids(trajectories, bird_ids):
    selected_trajectories = []
    for traj in trajectories:
        if len(traj) > 0 and len(traj[0]) > 0 and traj[0][0] in bird_ids:
            selected_trajectories.append(traj)
    return selected_trajectories


def calculate_acceleration_multi(selected_trajectories):
    acceleration_data = []

    for trajectory in selected_trajectories:
        acceleration_each = calculate_acceleration_fly_direction(trajectory)
        acceleration_data.append(acceleration_each)

    return acceleration_data


def calculate_acceleration(trajectory):
    acceleration = []
    for i in range(len(trajectory) - 1):
        trj_id = trajectory[i][0]
        frame = trajectory[i][1]
        xvel = trajectory[i][5]
        yvel = trajectory[i][6]
        zvel = trajectory[i][7]
        next_xvel = trajectory[i+1][5]
        next_yvel = trajectory[i+1][6]
        next_zvel = trajectory[i+1][7]
        xacc = (next_xvel - xvel)/(1/60)
        yacc = (next_yvel - yvel)/(1/60)
        zacc = (next_zvel - zvel)/(1/60)

        acceleration.append([trj_id, frame, xacc, yacc, zacc])
    return acceleration


def calculate_acceleration_fly_direction(trajectory):

    # Assuming equal time intervals between measurements
    time_interval = 1/60

    # Initialize a list to hold acceleration vectors in the custom coordinate system
    accelerations_custom_system = []

    for i in range(1, len(trajectory)):
        trj_id = trajectory[i][0]
        frame = trajectory[i][1]
        xpos, ypos, zpos, xvel, yvel, zvel = trajectory[i][2], trajectory[i][
            3], trajectory[i][4], trajectory[i][5], trajectory[i][6], trajectory[i][7]
        # Calculate the acceleration vector
        # Convert the inputs to numpy arrays for easy manipulation
        position = np.array([xpos, ypos, zpos]).T
        pre_position = np.array([trajectory[i - 1][2],
                                 trajectory[i - 1][3], trajectory[i - 1][4]]).T
        velocity = np.array([xvel, yvel, zvel]).T
        pre_velocity = np.array([trajectory[i - 1][5],
                                 trajectory[i - 1][6], trajectory[i - 1][7]]).T

        acceleration_vector = (
            velocity - pre_velocity) / time_interval

        # Define the custom coordinate system based on the y, z, and cross product for x
        y_direction = pre_velocity - pre_position
        z_direction = np.array([0, 0, 1])
        x_direction = np.cross(y_direction, z_direction)
        # Normalize the direction vectors
        x_direction = x_direction / np.linalg.norm(x_direction)
        y_direction = y_direction / np.linalg.norm(y_direction)
        z_direction = z_direction / np.linalg.norm(z_direction)

        # Create the transformation matrix
        transformation_matrix = np.array(
            [x_direction, y_direction, z_direction]).T

        # Transform the acceleration vector into the custom coordinate system
        acceleration_in_custom_system = np.dot(
            transformation_matrix, acceleration_vector)
        acceleration = np.insert(
            acceleration_in_custom_system, 0, [trj_id, frame])

        # Append the result to the list
        accelerations_custom_system.append(acceleration)

    return np.array(accelerations_custom_system)
