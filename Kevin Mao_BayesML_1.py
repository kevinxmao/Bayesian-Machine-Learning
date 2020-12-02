# Kevin Mao
# Project 1

import numpy as np
import statsmodels.api as sm
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import mean_squared_error

# generate list of bernoulli random variables
size = 200
mu = 0.5
Y = np.random.binomial(1, mu, size)

# Good parameters
a1 = 10
b1 = 10

# bad parameters
c1 = 1
d1 = 10


mu_ml = []
mu_bay_bad = []
mu_bay_good = []
mu_list = []
MSE_bay_good = []
MSE_bay_bad = []
MSE_ml = []

# Build a grid of 4 plots to show intermediate steps
fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(2, 2)
p1 = fig.add_subplot(gs[0])
p2 = fig.add_subplot(gs[1])
p3 = fig.add_subplot(gs[2])
p4 = fig.add_subplot(gs[3])

X = np.linspace(0,1, size)

# Initial distribution with good assumptions of hyperparameters
p1.plot(X, stats.beta(a1, b1).pdf(X), 'r')
p1.set(title='Prior Distribution (good parameters)')

# Iterate through the sample data
for i in range(1,size+1):

    # build sub list of sample observations
    subY = Y[0:i]

    # ML estimate
    mu_ml.append(subY.sum()/i)

    mu_list.append(mu)

    # update good hyperparameters
    a2 = a1 + subY.sum()
    b2 = b1 + size - subY.sum()

    # update bad hyperparameters
    c2 = c1 + subY.sum()
    d2 = d1 + size - subY.sum()

    # MAP/bayesian estimate of mean
    mu_bay_good.append(a2/(a2 + b2))
    mu_bay_bad.append(c2 / (c2 + d2))

    # calculate mean squared error for ML estimates and bayesian estimates
    MSE_bay_good.append(mean_squared_error(mu_list, mu_bay_good))
    MSE_bay_bad.append(mean_squared_error(mu_list, mu_bay_bad))
    MSE_ml.append(mean_squared_error(mu_list, mu_ml))

    # Plot intermediate posterior distribution
    if i == size/4:

        # Plot the analytic posterior
        p2.plot(X, stats.beta(a2, b2).pdf(X), 'g')

        p2.set(title='Posterior Distribution Intermediate 1')

    # Plot second intermediate posterior distribution
    if i == size/2:

        # Plot the analytic posterior
        p3.plot(X, stats.beta(a2, b2).pdf(X), 'g')

        p3.set(title='Posterior Distribution Intermediate 2')

# Posterior hyperparameters
a2 = a1 + Y.sum()
b2 = b1 + size - Y.sum()

# Plot the analytic posterior
p4.plot(X, stats.beta(a2, b2).pdf(X), 'g')

# Cleanup
p4.set(title='Posterior Distribution')

fig.legend([p1, p2, p3, p4],     # The line objects
           labels=['Conjugate Prior', 'Posterior'],   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )

# Plot mean squares error
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,size,size), MSE_bay_good, 'y')
ax.plot(np.linspace(1,size,size), MSE_bay_bad, 'r')
ax.plot(np.linspace(1,size,size), MSE_ml, 'b')
ax.legend(['Bayesian good','Bayesian bad', 'ML'])
ax.set(title='Mean Squared Error (good parameters)')

plt.show()

