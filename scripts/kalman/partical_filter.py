# coding: utf-8

import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Simulation parameters
N = 1000  # Number of particles
dt = 0.1  # Time step

# Initialize particle samples for position and velocity with normal distributions
xsamp = np.random.normal(0, 1, N)
vsamp = np.random.normal(0, 1, N)
wt = np.zeros((N,))

# Lists to hold the true state (XT), observation (ZT), estimates (XEST), and time (T)
XT = []
ZT = []
XEST = []
T = []

# Measurement and process noise variances
R = 1  # Variance of process noise
Q = 1  # Variance of observation noise

# Initial true state for position and velocity
x = 0
v = 1

# Simulation time variable
t = 0
while t < 10:
    # Predict step: update position and velocity of particles
    xsamp = xsamp + vsamp * dt
    vsamp = vsamp + np.random.normal(0, math.sqrt(R), N) * math.sqrt(dt)

    # Generate an observation from the true state with added noise
    z = x + random.gauss(0, math.sqrt(Q))

    # Update weights based on the observation
    # wt = np.exp(-np.square(z - xsamp) / (2 * Q))
    wt = np.exp(-np.square(z - xsamp) / (2 * Q)) / np.sqrt(2 * np.pi * Q)

    wsum = np.sum(wt)
    wt = wt / wsum

    # Estimate the state using the weighted mean of the particles
    xest = wt.dot(xsamp)
    vest = wt.dot(vsamp)

    # Store the true state, observation, estimate, and time
    XT.append(x)
    ZT.append(z)
    XEST.append(xest)
    T.append(t)

    # Resample particles based on the weights
    xv = [[x, v] for x, v in zip(xsamp, vsamp)]
    new_sample = np.array(random.choices(xv, k=N, weights=wt))
    xsamp = new_sample[:, 0]
    vsamp = new_sample[:, 1]

    # Update the true state with the process model
    x = x + v * dt
    v = v + random.gauss(0, math.sqrt(R)) * math.sqrt(dt)

    # Increment time
    t = t + dt

# Plot the true state, estimate, and observations over time
plt.plot(T, XT, color='blue', linewidth=1.0, label='State Model')
plt.plot(T, XEST, 'o', color='red', label='Estimate')
plt.plot(T, ZT, '*', color='green', linewidth=1.0, label='Observation')
plt.xlabel('Time')
plt.ylabel('Position')
plt.grid(True)
plt.legend()
plt.show()
