# Kevin Mao
# Project 1 Part 2

import numpy as np
import statsmodels.api as sm
import sympy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import mean_squared_error

# Simulate data
size = 200

# known variance and unknown mean
mu, var = 5, 2
Y = np.random.normal(mu, math.sqrt(var), size)

# Mu and Var Prior bad
mu_0_bad = 0
var_0_bad = 0.3

# Mu and var prior good
mu_0_good = 5
var_0_good = 2

mu_ml = []
mu_bay_bad = []
mu_bay_good = []
mu_list = []
MSE_bay_good = []
MSE_bay_bad = []
MSE_ml = []

fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(2, 2)
p1 = fig.add_subplot(gs[0])
p2 = fig.add_subplot(gs[1])
p3 = fig.add_subplot(gs[2])
p4 = fig.add_subplot(gs[3])

X = np.linspace(-10,10, size)

# Initial distribution
p1.plot(X, stats.norm.pdf(X, mu_0_good, math.sqrt(var_0_good)), 'r')
p1.set(title='Prior Distribution (good parameters')

for i in range(1,size+1):
    subY = Y[0:i]
    mu_ml.append(subY.sum()/i)
    mu_list.append(mu)

    N = len(subY)

    # update bayesian mean and variance
    mu_N_good = var/(N*var_0_good+var)*mu_0_good + N*var_0_good/(N*var_0_good+var)*mu_ml[i-1]
    var_N_good = 1/(1/var_0_good+N/var)

    mu_N_bad = var / (N * var_0_bad + var)*mu_0_bad + N * var_0_bad / (N * var_0_bad + var) * mu_ml[i - 1]
    var_N_bad = 1 / (1 / var_0_bad + N / var)

    # MAP/bayesian estimate of mean
    mu_bay_good.append(mu_N_good)
    mu_bay_bad.append(mu_N_bad)

    # calculate mean squared error for ML estimates and bayesian estimates
    MSE_bay_good.append(mean_squared_error(mu_list, mu_bay_good))
    MSE_bay_bad.append(mean_squared_error(mu_list, mu_bay_bad))
    MSE_ml.append(mean_squared_error(mu_list, mu_ml))

    # Plot intermediate posterior distribution
    if i == size/4:

        # Plot the analytic posterior
        p2.plot(X, stats.norm.pdf(X, mu_N_good, math.sqrt(var_N_good)), 'g')

        p2.set(title='Posterior Distribution Intermediate 1')

    # Plot second intermediate posterior distribution
    if i == size/2:

        # Plot the analytic posterior
        p3.plot(X, stats.norm.pdf(X, mu_N_good, math.sqrt(var_N_good)), 'g')

        p3.set(title='Posterior Distribution Intermediate 2')

#  Final posterior mean and variance
mu_N_good = var/(size*var_0_good+var)*mu_0_good + size*var_0_good/(size*var_0_good+var)*mu_ml[size-1]
var_N_good = 1/(1/var_0_good+size/var)

mu_N_bad = var/(size*var_0_bad+var)*mu_0_bad + size*var_0_bad/(size*var_0_bad+var)*mu_ml[size-1]
var_N_bad = 1/(1/var_0_bad+size/var)

# Plot the analytic posterior
p4.plot(X, stats.norm.pdf(X, mu_N_good, math.sqrt(var_N_good)), 'g')

# Cleanup
p4.set(title='Posterior Distribution')

fig.legend([p1, p2, p3, p4],     # The line objects
           labels=['Conjugate Prior', 'Posterior'],   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )

# Plot mean square error
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,size,size), MSE_bay_good, 'y')
ax.plot(np.linspace(1,size,size), MSE_bay_bad, 'r')
ax.plot(np.linspace(1,size,size), MSE_ml, 'b')
ax.legend(['Bayesian good','Bayesian bad', 'ML'])
ax.set(title='Mean Squared Error (good parameters)')

print(var_N_good)

plt.show()

