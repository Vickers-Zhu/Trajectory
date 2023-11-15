# Importing necessary libraries
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import threading


########################################################################
# Pre-defined parameters

# Specifying the target object for tracking
target = 0

# Initializing lists to store trajectory data (x, y, z coordinates) and frame numbers
xyz = []
frames = []

# Initializing starting position (x0, y0, z0) and velocity (vx0, vy0, vz0)
x0, y0, z0, vx0, vy0, vz0 = 0, 0, 0, 0, 0, 0

# Time interval between measurements (assuming 60Hz data rate)
dt = 1.0 / 60

# State transition matrix
# This matrix defines how the state evolves from one time step to the next
# without any control input. It includes the positions, velocities, and
# acceleration components for a 3D motion model.
F = np.array([[1, 0, 0, dt,  0,  0,  0,  0,  0],  # X position update
              [0, 1, 0,  0, dt,  0,  0,  0,  0],  # Y position update
              [0, 0, 1,  0,  0, dt,  0,  0,  0],  # Z position update
              [0, 0, 0,  1,  0,  0, dt,  0,  0],  # X velocity update
              [0, 0, 0,  0,  1,  0,  0, dt,  0],  # Y velocity update
              [0, 0, 0,  0,  0,  1,  0,  0, dt],  # Z velocity update
              # X acceleration (assumed constant)
              [0, 0, 0,  0,  0,  0,  1,  0,  0],
              # Y acceleration (assumed constant)
              [0, 0, 0,  0,  0,  0,  0,  1,  0],
              [0, 0, 0,  0,  0,  0,  0,  0,  1]])  # Z acceleration (assumed constant)

# Control input model matrix
# This matrix is used to apply the effect of control inputs to the state.
# In this case, it's configured to consider the acceleration as control input.
G = np.array([[0,  0,  0],  # No direct control on the X position
             [0,  0,  0],  # No direct control on the Y position
             [0,  0,  0],  # No direct control on the Z position
             [0,  0,  0],  # No direct control on the X velocity
             [0,  0,  0],  # No direct control on the Y velocity
             [0,  0,  0],  # No direct control on the Z velocity
             [math.sqrt(dt),  0,  0],  # Control impact on X acceleration
             [0,  math.sqrt(dt),  0],  # Control impact on Y acceleration
             [0,  0,  math.sqrt(dt)]])  # Control impact on Z acceleration

# Observation model matrix
# This matrix maps the true state space into the observed space.
# It's used to convert the predicted state into the same space or
# dimensions as the measurements.
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # X position observable
              [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Y position observable
              [0, 0, 1, 0, 0, 0, 0, 0, 0]])  # Z position observable

# Define initial state (position and velocity) and state covariance
X = np.zeros((9, 1))  # Initial state vector (all zeros)
XT = np.zeros((9, 1))  # Initial state estimate (all zeros)
S = np.zeros((9, 9))  # Initial state covariance matrix (all zeros)

# Define process noise covariance matrix Q and measurement noise covariance matrix R
# Q: Defines the amount of uncertainty in the process (model)
Q = np.array([[0.5, 0,   0],  # X position noise
              [0, 0.5,   0],  # Y position noise
              [0,   0, 0.5]])  # Z position noise

# R: Defines the amount of uncertainty in our observations (measurements)
R = np.array([[0.3, 0,   0],  # X position measurement noise
              [0, 0.3,   0],  # Y position measurement noise
              [0,   0, 0.3]])  # Z position measurement noise

# Set initial position and velocity estimates
XT[0, 0] = x0
XT[1, 0] = y0
XT[2, 0] = z0
XT[3, 0] = vx0
XT[4, 0] = vy0
XT[5, 0] = vz0
XT[6, 0] = 0
XT[7, 0] = 0
XT[8, 0] = 0

# Initialize state covariance matrix as identity matrix
S = np.eye(9)

# Lists to store results for plotting
TIM, PX, PY, PZ, VX, VY, VZ, VEL, AX, AY, AZ, ACC, AXERR = [
], [], [], [], [], [], [], [], [], [], [], [], []

########################################################################


def kalman3Dacc(lines):
    # Global variables
    global xyz, F, G, H, X, XT, S, Q, R
    xyz = []
    # Counter for processing lines
    cnt = 0
    for line in lines:
        cols = line.split()
        if len(cols) < 6:
            continue
        tid = int(cols[0])
        if tid != target:
            continue

        # Initialize start position and velocity
        if cnt == 0:
            x0, y0, z0 = float(cols[6]), float(cols[7]), float(cols[8])
        elif cnt == 1:
            vx0 = (float(cols[6]) - x0) / dt
            vy0 = (float(cols[7]) - y0) / dt
            vz0 = (float(cols[8]) - z0) / dt

        # Extracting time and xyz coordinates
        t = int(cols[1])
        x, y, z = float(cols[3]), float(cols[4]), float(cols[5])
        frames.append(t)
        xyz.append([x, y, z])
        cnt += 1

    # Convert xyz list to a NumPy array
    xyz = np.array(xyz)

    # Print the number of processed data points
    print(cnt)

    XT[0, 0] = x0
    XT[1, 0] = y0
    XT[2, 0] = z0
    XT[3, 0] = vx0
    XT[4, 0] = vy0
    XT[5, 0] = vz0
    XT[6, 0] = 0
    XT[7, 0] = 0
    XT[8, 0] = 0

    # Main loop for processing each frame
    for i in range(len(frames)):
        # Check for gaps in the data
        if i > 0 and frames[i] - frames[i - 1] > 1:
            print('GAP', i, frames[i])

        # Kalman Filter update steps
        D = np.linalg.pinv(H.dot(S.dot(H.T)) + R)
        K = S.dot(H.T).dot(D)

        # Observation
        Z = np.array([xyz[i, :]]).T

        # Update state estimate and state covariance
        X2 = XT + K.dot(Z - H.dot(XT))
        S2 = (np.eye(9) - K.dot(H)).dot(S)

        # Storing results for plotting
        TIM.append(frames[i])
        PX.append(XT[0, 0])
        PY.append(XT[1, 0])
        PZ.append(XT[2, 0])
        VX.append(XT[3, 0])
        VY.append(XT[4, 0])
        VZ.append(XT[5, 0])
        AX.append(XT[6, 0])
        AY.append(XT[7, 0])
        AZ.append(XT[8, 0])
        AXERR.append(math.sqrt(S[6, 6]))
        ACC.append(math.sqrt(XT[6, 0]**2 + XT[7, 0]**2 + XT[8, 0]**2))
        VEL.append(math.sqrt(XT[3, 0]**2 + XT[4, 0]**2 + XT[5, 0]**2))

        # Update state and covariance for next iteration
        XT = F.dot(X2)
        S = F.dot(S2.dot(F.T)) + G.dot(Q.dot(G.T))

        t = t + dt
    # Draw
    draw_kalman_acc()
    draw_kalman_3D()


def draw_kalman_3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Define a step size for downsampling
    step_size = 100  # Adjust this number to change the density of plotted points

    # Plot the trajectory from PX, PY, PZ
    ax.plot(PX, PY, PZ, color='blue', linewidth=1.5,
            label='Estimated (PX, PY, PZ)')

    # Plot the trajectory from xyz array
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='red',
            linewidth=1.5, linestyle=':', label='Observation (xyz array)')

    # Determine the maximum extent for the reference axes
    max_range = max(max(PX), max(PY), max(PZ), np.max(
        xyz[:, 0]), np.max(xyz[:, 1]), np.max(xyz[:, 2]))

    # Adding reference axes (X, Y, Z)
    ax.quiver(0, 0, 0, max_range, 0, 0, color='red',
              arrow_length_ratio=0.05)   # X-axis
    ax.quiver(0, 0, 0, 0, max_range, 0, color='green',
              arrow_length_ratio=0.05)  # Y-axis
    ax.quiver(0, 0, 0, 0, 0, max_range, color='blue',
              arrow_length_ratio=0.05)  # Z-axis

    # Adding labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory Plot')

    # Optionally setting the limits if necessary, adjust these as per your data range
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    plt.legend()
    plt.show()


def draw_kalman_acc():
    # Writing acceleration data to a file
    # with open('acc_data/' + str(target) + '.txt', 'w') as f:
    #     for i in range(len(frames)):
    #         print(frames[i], AX[i], AY[i], AZ[i], file=f)
    # Plotting the acceleration components
    plt.plot(TIM, AX, color='blue', linewidth=1.0, label='AX')
    plt.plot(TIM, AY, color='green', linewidth=1.0, label='AY')
    plt.plot(TIM, AZ, color='red', linewidth=1.0, label='AZ')
    plt.title('Trajectory #' + str(target))
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.grid(True)
    plt.legend()
    # plt.xlim(1800, 2700)
    plt.show()


if __name__ == "__main__":
    file = open("data/20201206-S8F1328E1#1S20.trj")
    data = file.readlines()
    file.close()
    kalman3Dacc(data)
