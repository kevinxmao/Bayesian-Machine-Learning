import numpy as np
import pandas as pd
import statsmodels.api as sm
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy import stats
from scipy.special import gamma
from sklearn.metrics import mean_squared_error

# Simulate data

size = 4000
mu = 0.5
Y = np.random.binomial(1, mu, size)

# Good parameters
ag1 = 10
bg1 = 10

mu_ml = []
mu_bay = []
mu_list = []
MSE_bay = []
MSE_ml = []

fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(2, 2)
p1 = fig.add_subplot(gs[0])
p2 = fig.add_subplot(gs[1])
p3 = fig.add_subplot(gs[2])
p4 = fig.add_subplot(gs[3])

X = np.linspace(0,1, 4000)

# Initial distribution
p1.plot(X, stats.beta(a1, b1).pdf(X), 'r')
p1.set(title='Prior Distribution (good parameters')

for i in range(1,size+1):
    subY = Y[0:i]
    mu_ml.append(subY.sum()/i)
    mu_list.append(mu)
    a2 = a1 + subY.sum()
    b2 = b1 + size - subY.sum()

    mu_bay.append(a2/(a2 + b2))

    MSE_bay.append(mean_squared_error(mu_list, mu_bay))
    MSE_ml.append(mean_squared_error(mu_list, mu_ml))

    if i == size/4:

        # Plot the analytic posterior
        p2.plot(X, stats.beta(a2, b2).pdf(X), 'g')

        # Plot the prior
        p2.plot(X, stats.beta(a1, b1).pdf(X), 'r')

        p2.set(title='Posterior Distribution Intermediate 1')

    if i == size/2:

        # Plot the analytic posterior
        p3.plot(X, stats.beta(a2, b2).pdf(X), 'g')

        # Plot the prior
        p3.plot(X, stats.beta(a1, b1).pdf(X), 'r')

        p3.set(title='Posterior Distribution Intermediate 2')


a2 = a1 + Y.sum()
b2 = b1 + size - Y.sum()

# Plot the analytic posterior
p4.plot(X, stats.beta(a2, b2).pdf(X), 'g')

# Plot the prior
p4.plot(X, stats.beta(a1, b1).pdf(X), 'r')

# Cleanup
p4.set(title='Posterior Distribution')
p4.legend(['Posterior', 'Conjugate Prior'])

# Plot mean square error
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,size,size), MSE_bay, 'y')
ax.plot(np.linspace(1,size,size), MSE_ml, 'b')
ax.legend(['Bayesian', 'ML'])
ax.set(title='Mean Squared Error (good parameters)')

plt.show()

