import random
import math
import matplotlib.pyplot as plt


def kalman_filter(x_values):
    # Initialization of lists to store results and other associated values
    T = []
    V = []
    XT, XTERR, ZT, SXX = [], [], [], []

    # Initial values for velocity, time, and covariance
    v = 1
    t = 0
    dt = 1/60
    xt, vt, sxx, sxv, svv = -1, 0, 1, 1, 1

    for x in x_values:
        R, Q = 1.0, 2.0  # Process and measurement noise covariances

        # Generate an observation of the state by adding noise to x
        zt = x + Q*random.gauss(0, math.sqrt(Q))

        # Kalman gains
        kx = sxx/(sxx+Q)
        kv = sxv/(sxx+Q)

        # Update estimates
        xt2 = xt + kx * (zt - xt)
        vt2 = vt + kv * (zt - xt)

        # Update covariance of estimation errors
        sxx2 = sxx - sxx**2/(sxx+Q)
        sxv2 = sxv - sxx*sxv/(sxx+Q)
        svv2 = svv - sxv**2/(sxx+Q)

        # Update state estimate based on velocity
        xt = xt2 + vt2 * dt
        vt = vt2

        # Store values
        T.append(t)
        XT.append(xt)
        ZT.append(zt)
        XTERR.append(math.sqrt(sxx2))
        SXX.append(sxx2)

        # Update covariance values
        sxx = sxx2 + sxv2*dt + svv2*dt**2
        sxv = sxv2 + svv2*dt
        svv = svv2 + R*dt

        # Update internal state with random acceleration
        at = random.gauss(0, math.sqrt(R))
        v = v + at*math.sqrt(dt)
        t = t + dt

    return T, XT, XTERR, ZT, SXX


def plot_kalman_results(T, x_values, XT, XTERR, ZT):
    plt.plot(T, x_values, color='blue', linewidth=7, label='State Model')
    plt.errorbar(T, XT, yerr=XTERR, capsize=0.01, fmt='o',
                 color='red', ecolor='orange', label='Estimate')
    plt.plot(T, ZT, '*', color='green', linewidth=0.1, label='Observation')
    plt.xlabel('T')
    plt.ylabel('X')
    plt.grid(True)
    plt.legend()
    plt.show()
