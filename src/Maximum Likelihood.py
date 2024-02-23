
from scipy.optimize import curve_fit
from scipy.stats import rayleigh
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt
import math
import matplotlib.colors as mcolors
num = 50
fig, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.6, 1]})
chi = 0
def distance_travel(x1, y1, x2, y2):
    conv = 0.1155  #10^-6 m
    dx = x1-x2
    dy = y1-y2
    dx = conv*dx
    dy = conv*dy
    return (dx**2 + dy**2)**0.5


file_names = ["bead1.txt", "bead2.txt", "bead3.txt", "bead4.txt",
              "bead5_.txt", "bead6.txt", "bead7.txt", "bead8.txt",
              "bead9.txt", "bead10.txt", "bead11.txt", "bead12.txt",
              "bead13.txt", "bead14.txt", "bead15.txt", "bead16.txt",
              "bead17.txt", "bead18.txt", "bead19.txt", "bead20.txt",
              "bead21.txt", "bead22.txt", "bead23.txt", "bead24.txt",
              "bead25.txt", "bead26.txt", "bead27.txt", "bead28.txt",
              "bead29.txt", "bead30.txt", "bead31.txt"]
d = []
for elem in file_names:
  dir = "/Users/yijiewang/Documents/WINTER 2024/PHY294/Thermal Motion/" + elem
  data = loadtxt(dir, usecols=(0, 1), skiprows=2, unpack=True)
  x_data = data[0, :]
  y_data = data[1, :]
  for i in range(len(x_data)-1):
     d.append(distance_travel(x_data[i], y_data[i],x_data[i+1], y_data[i+1]))
plt.figure(3)
plt.subplot(2, 1, 1)
plt.hist(d, bins=num, density=True, color = "#1f77b4", ec="black", lw=1)


'''
3. Distribution fitting using Scipy curve_fit()function
'''
hist, bin = np.histogram(d, bins=num, density=True)
start = (bin[0] + bin[1]) / 2
end = (bin[-1] + bin[-2]) / 2
bin = np.linspace(start, end, num=num)

def rayleigh(r, D):
   t = 0.5 #using t = 60s/120 = 0.5s
   return (r / (2*D*t)) * np.exp(-(r**2)/(4*D*t))

popt, pcov = curve_fit(rayleigh, bin, hist)
plt.plot(bin, rayleigh(bin, *popt), '-', label="Best Fit Curve", color="red", lw=1.5)
plt.legend()
ax1.legend(loc='upper right')
u_a=pcov[0,0]**(0.5)
print("D uncertianty", u_a, "\n")
#print("u_b", u_b, "\n")
y_perdict = rayleigh(bin, *popt)
corr_matrix = np.corrcoef(hist, y_perdict)
corr = corr_matrix[0,1]
R_sq = corr**2
print("Rsq", R_sq)
deltay = 1

for i in range(len(hist)):
   deltay = (deltay**2+0.1**2)**0.5
   chi = chi + ((hist[i] - y_perdict[i])**2)/(deltay)

print(chi)
chi = chi/((deltay**2) * (len(y_perdict) - 1))
print("Chi2",chi)

'''
4. Calculate K - Rayleigh Distribution
'''
D = popt[0]
T = 283.15 #20
eta = 1.21899
r = 0.95
gamma = 6 * math.pi * eta * r
K = D * gamma / T
K = K * 10**(-21)

print("K = ", K)
print("D = ", D)

k_accepted = 1.38*(10**(-23))
percent_diff = (k_accepted - K)/k_accepted * 100
print("\nThe percent difference is: ")
print(percent_diff, "%")

'''
residuals = hist - rayleigh(bin, *popt)
plt.subplot(2, 1, 2)
plt.scatter(bin, residuals, color='black')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals', fontsize = 16)
plt.xlabel('distance [μm]', fontsize = 14)
plt.ylabel('probablity', fontsize = 14)
'''

'''
4. Most likely K
'''
max_likelihood = 0
for elem in d:
   max_likelihood = max_likelihood + elem**2

max_likelihood = max_likelihood/(2*len(d))
D_maxlikelihood = max_likelihood/(2*0.5)

print('D_maxlikelihood', D_maxlikelihood)
D = D_maxlikelihood
K = D * gamma / T
K = K * 10**(-21)


print("K", K)
percent_diff = (k_accepted - K)/k_accepted * 100
print("\nThe percent difference is: ")
print(percent_diff, "%")

plt.plot(bin, rayleigh(bin, D), '--', label="Maximum Likelihood", color="blue", lw=1.5)
plt.legend()

corr_matrix = np.corrcoef(hist, rayleigh(bin, D))
corr = corr_matrix[0,1]
R_sq = corr**2
print("Rsq3", R_sq)
deltay = 1


plt.title('Histogram of Brownian Motion Fitted with Rayleigh Distribution and Maximum Likelihood')
plt.xlabel('Distance [μm]')
plt.ylabel('Probablity')

residuals = hist - rayleigh(bin, 0.16443376002930982)
print(residuals)
plt.subplot(2, 1, 2)
plt.scatter(bin, residuals, color='#1f77b4')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals of Rayleigh Distribution with Maximum Likelihood')
plt.xlabel('Distance [μm]')
plt.ylabel('Probablity')


plt.tight_layout()
plt.show()