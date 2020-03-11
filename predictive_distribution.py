# with help from Nick Triano for remaking 3.8
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
    noise_dev = 0.2
    beta = (1 / noise_dev) ** 2
    noise = np.random.normal(0, noise_dev, size)    # gaussian noise
    x = np.random.uniform(0, 1, size)   # generate uniform random variables
    x_sort = np.sort(x) # a sorted x array for later use
    t = np.array(np.sin(2*np.pi*x) + noise)     # create targets in data set
    t_sort = np.array(np.sin(2 * np.pi * x_sort) + noise)   # create targets on sorted x array
    return x, t, x_sort, t_sort, beta


def main():
    # generate sample data set
    size = 100
    x, t, x_sort, t_sort, beta = sampleGenerator(size)

    # create a Phi matrix of zeros with size 9 x sample size
    Phi = np.zeros((9, len(x)))
    Phi_sort = np.zeros((9, len(x)))    # a sorted Phi matrix for plotting use
    s = 0.5     # choose a good estimate of spacial

    # prior mean and covariance matrix
    s0 = np.linalg.inv(np.eye(9) * 0.2)
    m0 = np.zeros((9, 1))

    fig, axs = plt.subplots(2, 2)

    # in the following loop, Phi is created with equation 3.4
    for i in range(0, len(x)):
        for j in range(0, 9):
            mu = (j-4)/4    # a good estimate of mu
            Phi[j, i] = np.exp(-(x[i]-mu)**2/2/s**2)    # basis function equation 3.4
            Phi_sort[j, i] = np.exp(-(x_sort[i] - mu) ** 2 / 2 / s ** 2)

    for N in range(0, 25):
        # create iota matrix from the basis functions
        iota = np.array([Phi[:, N]])

        # posterior covariance equation 3.51
        sn = np.linalg.inv(s0) + beta*np.matmul(iota.T, iota)
        sn = np.linalg.inv(sn)

        # posterior mean equation 3.50
        mn = np.dot(sn, np.matmul(np.linalg.inv(s0), m0) + beta*iota.T*t[N])

        # create a 1-D array of mean of predictive distribution
        # equation 3.58
        m_x = np.squeeze(np.matmul(mn.T, Phi_sort))

        # calculate sigma from iota, Phi, and posterior covariance
        sigma_x = [0]*len(x)
        for n in range(0, Phi.shape[1]):
            # a column vector of basis function
            phi = np.array([Phi_sort[:, n]]).T

            sigma_x[n] = np.squeeze((1/beta + np.dot(np.dot(phi.T, sn), phi))**(1/2))

        s0 = np.copy(sn)
        m0 = np.copy(mn)

        # plot the intermediate steps and results when sample size is 1, 2, 4, 25
        if N in [0, 1, 3, 24]:
            if N == 0:
                r = 0
                c = 0
            elif N == 1:
                r = 0
                c = 1
            elif N == 3:
                r = 1
                c = 0
            elif N == 24:
                r = 1
                c = 1

            axs[r, c].plot(x_sort, m_x, color="red")

            X = np.linspace(0,1,100)
            axs[r, c].plot(X, np.sin(2 * np.pi * X), '-g')

            axs[r, c].scatter(x[:N+1], t[:N+1], marker='o', facecolors="none", edgecolors='blue')
            axs[r, c].fill_between(x_sort, m_x+sigma_x, m_x-sigma_x, alpha=0.3, color='pink')
            axs[r, c].set_xlim([0, 1])
            axs[r, c].set_ylim([-1, 1])

    plt.show()


if __name__ == '__main__':
    main()

