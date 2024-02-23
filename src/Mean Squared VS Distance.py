from scipy.optimize import curve_fit
from scipy.stats import rayleigh
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt
import math

def dist(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def rayleigh_dist(r, sigma):
    return (r / (sigma**2)) * np.exp(-(r**2) / (2 * sigma**2))

def main():
    step_size = []

    for i in range(1, 30):
        if i == 5:
            continue

        filename=f"/Users/yijiewang/Documents/WINTER 2024/PHY294/Thermal Motion/bead{i}.txt"

        data=loadtxt(filename, usecols=(0,1), skiprows=2, unpack=True)

        xdata = data[0]
        ydata = data[1]
        for j in range(len(xdata)-1):
            step_size.append(dist(xdata[j], xdata[j+1], ydata[j], ydata[j+1]))
    print(len(step_size))
    step_size = np.array(step_size)
    # hist, bins = np.histogram(step_size, bins=50, density=True)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_width = 0.1  # Set the desired bin width
    min_edge = min(step_size)  # Find the minimum value in the data
    max_edge = max(step_size)  # Find the maximum value in the data
    bins = np.arange(min_edge, max_edge + bin_width, bin_width)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    hist, bins = np.histogram(step_size, bins=bins, density=True)
    # Create a range of bin edges from min to max with the specified width

    plt.hist(step_size, bins=bins, density=True, alpha=0.6, color='g')

    init_guess = np.sqrt(np.mean(step_size ** 2) / 2)

    params, covariance = curve_fit(rayleigh_dist, bin_centers, hist, p0=[init_guess])
    sigma_fit, sigma_fit_error = params[0], np.sqrt(covariance[0, 0])
    print(f'Sigma value from Rayleigh distribution fit: {sigma_fit} micrometers')
    print(f'Uncertainty for sigma: {sigma_fit_error} micrometers')

    # Plot the fitted Rayleigh distribution curve
    r = np.linspace(0, data.max(), 100)
    plt.plot(r, rayleigh_dist(r, sigma_fit), 'r-', label='Fitted Rayleigh distribution')

    plt.axis([0, 20, 0, 0.2])  # Set x-axis from 0 to 6 and y-axis from 0 to 30

    plt.show()

main()
