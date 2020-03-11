# generate graph 3.7
# Kevin Mao

import numpy as np
import statsmodels.api as sm
import sympy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.stats import multivariate_normal

# Create a method that generates data set
def sampleGenerator(size):
    a_0 = -0.3  # true weight
    a_1 = 0.5   # true weight
    noise_dev = 0.2
    alpha = 2.0
    beta = (1/noise_dev)**2
    x = np.random.uniform(-1, 1, size)  # generate uniform random variables
    noise = np.random.normal(0, noise_dev, size)    # gaussian noise
    t = np.array(a_0 + a_1 * x + noise)     # target
    return x, t, alpha, beta


def main():
    # Generate sample data set
    size = 300
    x, t, alpha, beta = sampleGenerator(size)

    # Create a gridspec of 12 subplots
    fig = plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(4, 3)
    l1 = fig.add_subplot(gs[0])
    p1 = fig.add_subplot(gs[1], autoscale_on = False)
    ds1 = fig.add_subplot(gs[2])
    l2 = fig.add_subplot(gs[3])
    p2 = fig.add_subplot(gs[4])
    ds2 = fig.add_subplot(gs[5])
    l3 = fig.add_subplot(gs[6])
    p3 = fig.add_subplot(gs[7])
    ds3 = fig.add_subplot(gs[8])
    l4 = fig.add_subplot(gs[9])
    p4 = fig.add_subplot(gs[10])
    ds4 = fig.add_subplot(gs[11])

    # first row has no likelihood, axis are turned off
    l1.axis("off")
    l1.set_title("likelihood")

    # Create an mgrid of range -1 to 1 for both x and y
    X, Y = np.mgrid[-1:1.1:0.1, -1:1.1:0.1]
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X;  pos[:, :, 1] = Y
    m0 = [0, 0]
    s0 = (alpha**-1)*np.eye(2)

    # weights are created from a multivariate normal distribution with prior mean and covariance matrix
    w = multivariate_normal(m0, s0)

    # plot prior distribution of two weights
    p1.contourf(X, Y, w.pdf(pos), levels=150, cmap='jet')
    p1.set_xlim([-1, 1])
    p1.set_ylim([-1, 1])
    p1.set_title("prior/posterior")

    # in the for loop, a random sample of weights are drawn from weight matrix and 6 lines are drawn
    for i in range(0, 6):
        prior = w.rvs()     # draw random sample from the weight distribution
        ds1.plot(X, prior[0]+prior[1]*X, '-r')
        ds1.set_xlim([-1, 1])
        ds1.set_ylim([-1, 1])
        ds1.set_title("data space")

    # the following loop iterates through all the data and updates linear regression model
    for num, val in enumerate(x, start=1):

        # observation of data is stored and reshaped into a 1xn array
        obs = np.array(x[:num])
        obs = np.reshape(obs, (1, num))

        # create an iota matrix of size nx2 with basis functions
        iota = np.concatenate((np.ones((num, 1)), obs.transpose()), axis=1)

        # update new covariance matrix through sample
        sN = np.linalg.inv(alpha*np.eye(2)+beta*np.dot(iota.transpose(), iota))

        # update new mean matrix through sample
        mN = np.dot(np.dot(beta*sN, iota.transpose()), t[:num].transpose())
        # reshape mean into a 1-D array
        mN = np.reshape(mN, (2,))

        # update new weight distribution
        w = multivariate_normal(mN, sN)

        # update  likelihood function and create a normal distribution on sample data
        w0, w1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        likelihood = norm(w0 + w1 * val, beta ** (-1/2))

        if num == 1:

            # plot likelihood function
            l2.contourf(w0, w1, likelihood.pdf(t[num-1]), levels=150, cmap='jet')
            l2.set_xlim([-1, 1])
            l2.set_ylim([-1, 1])
            l2.scatter(-0.3, 0.5, marker='+', color='white')

            # plot posterior
            p2.contourf(X, Y, w.pdf(pos), levels=150, cmap='jet')
            p2.set_xlim([-1, 1])
            p2.set_ylim([-1, 1])
            p2.scatter(-0.3, 0.5, marker='+', color='white')

            # in data space, plot the points in the sample
            ds2.scatter(val, t[num-1], marker='o', facecolors="none", edgecolors='blue')
            ds3.scatter(val, t[num - 1], marker='o', facecolors="none", edgecolors='blue')

            # in the for loop, a random sample of weights are drawn from weight matrix and 6 lines are drawn
            for i in range(0, 6):
                p = w.rvs()
                ds2.plot(X, p[0] + p[1] * X, '-r')
                ds2.set_xlim([-1, 1])
                ds2.set_ylim([-1, 1])

        if num == 2:
            l3.contourf(w0, w1, likelihood.pdf(t[num-1]), levels=150, cmap='jet')
            l3.set_xlim([-1, 1])
            l3.set_ylim([-1, 1])
            l3.scatter(-0.3, 0.5, marker='+', color='white')

            p3.contourf(X, Y, w.pdf(pos), levels=150, cmap='jet')
            p3.set_xlim([-1, 1])
            p3.set_ylim([-1, 1])
            p3.scatter(-0.3, 0.5, marker='+', color='white')

            ds3.scatter(x[:num], t[:num], marker='o', facecolors="none", edgecolors='blue')

            for i in range(0, 6):
                p = w.rvs()
                ds3.plot(X, p[0] + p[1] * X, '-r')
                ds3.set_xlim([-1, 1])
                ds3.set_ylim([-1, 1])

        if num == 20:
            l4.contourf(w0, w1, likelihood.pdf(t[num-1]), levels=150, cmap='jet')
            l4.set_xlim([-1, 1])
            l4.set_ylim([-1, 1])
            l4.scatter(-0.3, 0.5, marker='+', color='white')

            p4.contourf(X, Y, w.pdf(pos), levels=150, cmap='jet')
            p4.set_xlim([-1, 1])
            p4.set_ylim([-1, 1])
            p4.scatter(-0.3, 0.5, marker='+', color='white')

            ds4.scatter(x[:num], t[:num], marker='o', facecolors="none", edgecolors='blue')

            for i in range(0, 6):
                p = w.rvs()
                ds4.plot(X, p[0] + p[1] * X, '-r')
                ds4.set_xlim([-1, 1])
                ds4.set_ylim([-1, 1])

    plt.show()


if __name__ == '__main__':
    main()

