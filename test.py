# coding: utf-8

import random
import math
import matplotlib.pyplot as plt

# Initialization of lists to store state, time, and other associated values
X = []  # To store actual state values
# To store actual velocities (though this list is initialized, it's not used in the given code)
V = []
T = []  # Time stamps

XT = []  # To store estimated state values
XTERR = []  # To store estimation errors (standard deviation)
ZT = []  # Observations of the state

SXX = []  # Variance of the state estimation error

# Initial values for state, velocity, and time
x = 0
v = 1
t = 0
dt = 0.1  # Time step

# Initial estimates and covariance values for the state and velocity
xt = -1
vt = 0
sxx = 1
sxv = 1
svv = 1

# Start of Kalman filtering process
while t < 10:
    R = 1.0  # Process noise covariance for acceleration
    Q = 2.0  # Measurement noise covariance

    # Generate an observation of the state by adding noise to the state
    zt = x + Q*random.gauss(0, math.sqrt(Q))

    # Compute Kalman gains for state and velocity
    kx = sxx/(sxx+Q)
    kv = sxv/(sxx+Q)

    # Update the state and velocity estimates using the Kalman gains
    xt2 = xt + kx * (zt - xt)
    vt2 = vt + kv * (zt - xt)

    # Update the covariance of estimation errors
    sxx2 = sxx - sxx**2/(sxx+Q)
    sxv2 = sxv - sxx*sxv/(sxx+Q)
    svv2 = svv - sxv**2/(sxx+Q)

    # Further update the state estimate based on the velocity estimate
    xt = xt2 + vt2 * dt
    vt = vt2

    # Store values in their respective lists for later plotting
    T.append(t)
    X.append(x)
    XT.append(xt)
    ZT.append(zt)
    XTERR.append(math.sqrt(sxx2))
    SXX.append(sxx2)

    # Update the covariance values for next iteration
    sxx = sxx2 + sxv2*dt + svv2*dt**2
    sxv = sxv2 + svv2*dt
    svv = svv2 + R*dt

    # Update the internal state: Add a random acceleration to the velocity
    at = random.gauss(0, math.sqrt(R))
    x = x + v * dt
    v = v + at*math.sqrt(dt)
    t = t + dt

# Plot the true state, observations, and estimates with error bars
plt.plot(T, X, color='blue', linewidth=1.0, label='State Model')
plt.errorbar(T, XT, yerr=XTERR, capsize=5, fmt='o',
             color='red', ecolor='orange', label='Estimate')
plt.plot(T, ZT, '*', color='green', linewidth=1.0, label='Observation')
plt.xlabel('T')
plt.ylabel('X')
plt.grid(True)
plt.legend()
plt.show()
