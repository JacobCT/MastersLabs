# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:27:25 2022

@author: jacob
"""
import numpy as np
from scipy.special import jn_zeros, jnp_zeros

#Pre-Lab Exercise:
#Given variables
d = 0.0785 #d = 78.5 #mm
l = 0.02 #20 #(mm)
c = 2.99792458e8#2.99792458e11 #mm/s
p=0

#d/l squared
dl_square = (d/l)**2
print(dl_square)

#Value determined from graph
value = np.array([0.5e17, 1.3e17, 2.4e17, 2.8e17, 3.8e17, 3.7e17, 4e17, 4.2e17, 4.5e17, 4.8e17 ])
#Convert to a frequency
freq = np.sqrt(value)/d

#These values might not be right, it would be important to review them. Instead we use the code below this section
#jdashmn = np.array([2.40482, 3.83170, 5.52007, 7.01558, 8.65372, 10.17346, 11.79153, 13.32369, 14.93091, 16.47063])
jdashmn = np.array([3.83170, 3.83171, 7.01558, 7.01559, 10.17346, 10.17347, 13.32369, 13.32369, 16.47063, 16.47063])
#jdashmn = np.array([1.84118, 2.40482, 7.01558, 7.01559, 10.17346, 10.17347, 13.32369, 13.32369, 16.47063, 16.47063])
#jdashmn = np.array([1.84118, 2.400482, 5.52007, 5.33144, 8.53632, 8.65372, 11.70600, 11.79153, 14.86359, 14.93091])
#Given equation
dv_sqr = ((c*jdashmn)/np.pi)**2  + ((c/2)**2 * p**2 * dl_square)
#Convert to a frequency
freq2 = np.sqrt(dv_sqr)/d

#######
#This block of code was created by Keito Watanabe and Paarth Thakkar and gives the right values.
#Redefine the variables
d = 78.5e-3 #m
l = 20e-3 #m
c = 3e8  # m

#Function to find the rest frequencies
def res_freq(m, n, p, mode="TM", Nzeros=5):
    jmn = jn_zeros(m,Nzeros)[n-1] if mode == "TM" else jnp_zeros(m,Nzeros)[n-1]
    return np.sqrt((c * jmn / np.pi)**2. + (c * p * d / (2 * l))**2) / d

#Ten lowest eigenmodes, listing (m,n,p,mode)
modes = [(0,1,0,"TM"), (1,1,0,"TM"), (2,1,0,"TM"), (0,2,0,"TM"),(1,1,1,"TE"), (3,1,0,"TM"),(0,1,1,"TM"),(2,1,1,"TE"),(1,2,0,"TM"),(1,1,1,"TM"),(0,1,1,"TE")]

freqs = []

for mode in modes:
    freq = res_freq(*mode)
    freqs.append(freq)
    print("m,n,p: ({0}, {1}, {2}), mode:{3}, freq:{4:.4f}GHz".format(*mode, freq*1e-9))



########

#Data Analysis
#For this part, we select the right variables for which to run the code

#Cavity1
#rho = 598.1e-3 #peak1
#rho = 72.825e-3 #peak2
#rho = 805.45e-3 #peak3
#rho = 338e-3 #peak4
#rho = 745e-3 #peak5
#omega_0 = 2.99225e9
#omega_0 = 4.539375e9
#omega_0 = 6.215e9
#omega_0 = 6.9409375e9
#omega_0 = 7.75875e9
#omega_H = (1.33125e6)*2
#omega_H = (omega_0 - 4.53496875e9)*2
#omega_H = (omega_0 - 6.2125e9)*2
#omega_H = (4.0625e6)*2
#omega_H = (omega_0 - 7.7549375e9)*2
#Vector Measurement Uncalibrated
#d = 475.4e-3
#rho = d/(781.2e-3) #475e-3
#omega_0 = 2.992125e9
#omega_H = (omega_0 - 2.993375e9)*2
#Vector Measurement Calibrated
#d = 3493e-3
#rho = d/(12.44e-3) #3493e-3
#omega_0 = 2.99225e9
#omega_H = (omega_0 - 2.99325625e9)*2

#Cavity2
rho = 742e-3 #peak1
#rho = 811e-3 #peak2
#rho = 928e-3 #peak3
#rho = 649e-3 #peak4
#rho = 676e-3 #peak5
omega_0 = 2.9867e9
#omega_0 = 4.49675e9
#omega_0 = 4.5331875e9
#omega_0 = 6.2075e9
#omega_0 = 6.924375e9
omega_H = (2.0e6)*2
#omega_H = (1.5e6)*2
#omega_H = (1.640625e6)*2
#omega_H = (328.125e3)*2
#omega_H = (10.625e6)*2

#Cavity3
#rho = 280e-3

#Calculate a value for Kappa. One value larger than zero and the other value smaller than zero
kappalarge = (1+np.absolute(rho))/(1-np.absolute(rho))
kappasmall = (1-np.absolute(rho))/(1+np.absolute(rho))

#Calculate values for the reflection coefficient at FWHM 
rho_fwhm1 = np.sqrt(kappalarge**2 + 1)/ (kappalarge + 1)
rho_fwhm2 = np.sqrt(kappasmall**2 + 1)/ (kappasmall + 1)

#Calculate the quality factor 
Q = omega_0 / omega_H
#Calculate the unloaded quality factor 
Q_0 = ((omega_0 / omega_H)*(1+kappasmall))
#Calculate the external quality factor 
Q_ext = Q_0/kappasmall
#Print the values for each quality factor
print("Q: " + str(Q))
print("Q_0: " + str(Q_0))
print("Q_ext: " + str(Q_ext))

#Input power
P_0 = 100000 #Megawatt
#Calculate the standing wave ratio
swr = (1 + np.absolute(rho))/(1 - np.absolute(rho))
#Calculate the power loss
Power = ((4*swr)/(1 + swr)**2)*P_0
#Print the calculated values for the standing wave ration and power loss
print("SWR: " + str(swr))
print("Power: " + str(Power))
