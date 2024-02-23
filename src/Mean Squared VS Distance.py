from scipy.optimize import curve_fit
from scipy.stats import rayleigh
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt
import math
import matplotlib.colors as mcolors

file_names = ["/Users/yijiewang/Documents/WINTER 2024/PHY294/Thermal Motion/bead1.txt"]
t = []
d = []
data = loadtxt(file_names[0], usecols=(0, 1), skiprows=2, unpack=True)
x_data = data[0, :]
y_data = data[1, :]
def distance_travelled(x1, y1, x2, y2):
   conv = 0.1155  #10^-6 m
   dx = x1-x2
   dy = y1-y2
   dx = conv*dx
   dy = conv*dy
   return (dx**2 + dy**2)**0.5

def dist(x1, y1, x2, y2):
   conv = 0.1155  #10^-6 m
   dx = x1-x2
   dy = y1-y2
   dx = conv*dx
   dy = conv*dy
   return (dx**2 + dy**2)**0.5
for i in range(len(x_data)-1):
   if i == 0:
      t.append(0)
      d.append(dist(x_data[i], y_data[i],x_data[i+1], y_data[i+1]))
      dist(x_data[i], y_data[i],x_data[i+1], y_data[i+1])
   else:
      d.append(dist(x_data[i], y_data[i],x_data[i+1], y_data[i+1]) + d[-1])
      dist(x_data[i], y_data[i],x_data[i+1], y_data[i+1])
      t.append(t[-1] + 0.5)

xdata = np.array(t)
xerror = 0.03
yerror = 0.1
ydata = np.array(d)


def my_func(t, m):
    return t*m 

popt, pcov = curve_fit(my_func, xdata, ydata)
a=popt[0]
u_a = pcov[0, 0]
print(a, u_a)

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = 10, 9

start = min(xdata)
stop = max(xdata)
xs = np.arange(start,stop,(stop-start)/1000)
curve = my_func(xs, *popt)
fig, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.6, 1]})

ax1.errorbar(xdata, ydata, yerr=yerror, xerr=xerror, fmt=".", label="data", color="blue")
ax1.plot(xs, curve, label="best fit", color="black")
ax1.legend(loc='upper right')

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Mean Squared Distance [μm]")
ax1.set_title("Step Size Versus Time plot for a Sample Bead Motion")

residual = ydata - my_func(xdata, popt[0])
ax2.errorbar(xdata, residual, yerr=yerror, xerr=xerror, fmt=".", color="blue")
# Plot the residuals with error bars.

y_perdict = my_func(xdata, a)
corr_matrix = np.corrcoef(ydata, y_perdict)
corr = corr_matrix[0,1]
R_sq = corr**2
print("R_sq1", R_sq)

deltay = 0.1
chi = 0
for i in range(len(y_perdict)):
   deltay = (deltay**2+0.1**2)**0.5
   chi = chi + ((ydata[i] - y_perdict[i])**2)/(deltay)

chi = chi/((deltay**2) * (len(y_perdict) - 1))
print("Chi1",chi)


ax2.axhline(y=0, color="black")

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Mean Squared Distance [μm]")
ax2.set_title("Residuals")

plt.tight_layout()
plt.show()
