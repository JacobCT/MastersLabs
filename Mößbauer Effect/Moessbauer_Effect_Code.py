# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:37:10 2022

@author: jacob
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from uncertainties import unumpy,ufloat

#Define a Lorentz function
def lorentz_func(nu, a, tau, nu_0, b):
    return a* ( (tau/2)**2 / ( (nu-nu_0)**2 + (tau/2)**2 ) ) + b

#Define a combined lorentz function with 6 peaks
def lorentz_func_comb (nu,
                       a_n3, a_n2, a_n1, a_p1, a_p2, a_p3,
                       tau_n3, tau_n2, tau_n1, tau_p1, tau_p2, tau_p3,
                       nu_0_n3, nu_0_n2, nu_0_n1, nu_0_p1, nu_0_p2, nu_0_p3,
                       b):
    return lorentz_func(nu, a_n3, tau_n3, nu_0_n3, b) + lorentz_func(nu, a_n2, tau_n2, nu_0_n2, b) + lorentz_func(nu, a_n1, tau_n1, nu_0_n1, b) + lorentz_func(nu, a_p2, tau_p2, nu_0_p2, b) + lorentz_func(nu, a_p3, tau_p3, nu_0_p3, b) + lorentz_func(nu, a_p1, tau_p1, nu_0_p1, b)


#Convert from mm/s to neV
def E(v):
   return ((14.4*(1000000000000))/2.998e+11)*v #((14.4*(10000000000))/2.998e+11)*v

#--------------------------------------
#First Part
#Create figure
plt.figure(figsize=(7, 5), dpi=100)

#Counts per 10s
countsold = np.array([2882, 2482, 2275, 2550, 2500, 2206, 2128, 1861, 1777, 1630, 1658, 1492, 1432, 1376, 1313, 1216, 1161, 1071, 1037, 997, 932, 904, 892, 767, 726, 689, 712, 623, 637, 600, 591, 657, 704, 728, 842, 972, 1076, 1293, 1371, 1486, 1570, 1612, 1718, 1701, 1592, 1542, 1331, 1221, 1067, 856, 765, 691, 532, 451, 331, 330, 264, 216, 183, 150, 154, 144, 101, 112, 112, 100, 98, 106, 87, 98, 84, 79, 97, 94, 107, 110, 85, 103, 98, 97, 109, 103, 97, 106, 89, 85, 92, 82, 93, 91, 76, 88, 90, 92, 99, 98, 92, 88, 71, 95, 95, 118, 109, 140, 181, 198, 225, 248, 327, 357, 396, 441, 474, 564, 639, 639, 658, 654, 676, 687, 627, 591, 555, 462, 423, 394, 330, 315, 264, 230, 202, 182, 155, 123, 121, 112, 100, 86])
counts = np.array([2882, 2482, 2275, 2550, 2500, 2206, 2128, 1861, 1777, 1630, 1658, 1492, 1432, 1376, 1313, 1216, 1161, 1071, 1037, 997, 932, 904, 892, 767, 726, 689, 712, 623, 637, 600, 591, 657, 704, 728, 842, 972, 1076, 1293, 1371, 1486, 1570, 1612, 1718, 1701, 1592, 1542, 1331, 1221, 1067, 856, 765, 691, 532, 451, 331, 330, 264, 216, 183, 150, 154, 144, 101, 112, 100, 98, 106, 87, 98, 84, 79, 97, 94, 107, 110, 85, 103, 98, 97, 109, 103, 97, 106, 89, 85, 92, 82, 93, 91, 76, 88, 90, 92, 99, 98, 92, 88, 71, 95, 118, 109, 140, 181, 198, 225, 248, 327, 357, 396, 441, 474, 564, 639, 658, 654, 676, 687, 627, 591, 555, 462, 423, 394, 330, 315, 264, 230, 202, 182, 155, 123, 121, 112, 100, 94, 78, 73, 81, 82, 68, 70, 86, 71, 88, 71, 90, 80, 74, 71, 96, 84, 77, 86, 97, 80, 102, 107, 115, 110, 108, 140, 132, 164, 162, 139, 166, 161, 164, 171, 174, 149, 138, 170, 152, 141, 118, 116, 140, 111, 100, 126, 113, 98, 110, 101, 87, 95, 71, 81, 86, 73, 77, 78, 93]) #modification is before 100.

#units of mV
volts = np.arange(320,4200,20)

time = 10
#Plot the data
yerr=2
plt.plot(volts, counts/time, '.',linestyle='dashed',color='b',label='Data')
plt.errorbar(volts,counts/time,yerr,label='error=$\pm 2$ mV')
plt.axvline(x = 2200, color = 'black')
plt.axvline(x = 3000, color = 'r')
plt.text(2300,170,'Mößbauer \nSpectrum')
plt.xlabel('Voltage (mV)')
plt.ylabel('Photon Count ($s^{-1}$)')
plt.legend()
plt.grid()
plt.show()

#-----------------------
#Second PArt
#Number of passes
n_pass_lr = np.array([5,5,8,8,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,2,1]) # number of passes
n_pass_rl = np.array([5,5,8,8,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,2,1])
#The number on the motor
motor_n = np.array([2.2,2.1,2,1.9,1.85,1.8,1.75,1.73,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.85,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.18,0.15]) #Motor reading on the dial (human_error = 0.01)

#Positive side of the graph:
time_lr = np.array([1890,1977,3315,3488,2236,2293,2360,2381,2425,2571,2741,2931,3140,3387,3670,4000,2803,2967,3361,3889,4586,5625,7164,9869,7296,4227])
counts_lr = np.array([3760,3847,6519,6466,4000,3878,3858,4123,4219,4765,5283,5780,6137,6450,6442,6616,5261,5630,6625,7701,9076,10805,13208,18360,13763,8184])

#Negative side of the graph:
time_rl = np.array([2005,2100,3535,3730,2395,2460,2538,2561,2611,2783,2984,3209,3465,3766,4120,4537,3253,3474,4031,4816,5931,7793,11101,19310,15922,11392])
counts_rl = np.array([3968,4064,6652,6455,4021,4074,4520,4648,4810,5347,5763,6107,6586,7096,7202,8112,6326,6907,8002,9547,11412,14778,19807,36946,30811,22482])

#Calibrate time
time_lr_cal = (time_lr/100)
time_rl_cal = (time_rl/100)

#Convert to velocities
vel_target_lr = 100*(25.1*n_pass_lr)/time_lr
vel_target_rl = -100*((25.1*n_pass_rl)/time_rl)

vel_target_lr_rev = vel_target_lr[::-1] #Values flipped to be plotted in the last plot where entire spectrum has to be shown.

#Calibrate the counts
counts_lr_cal = (counts_lr)/time_lr_cal
counts_rl_cal = (counts_rl)/time_rl_cal

counts_lr_cal_rev = counts_lr_cal[::-1] #Values flipped to be plotted in the last plot where entire spectrum has to be shown.

#Append rl and lr data
time_cal = np.append(time_rl_cal, time_lr_cal)
n_pass = np.append(n_pass_rl, n_pass_lr)
xdat = np.append(vel_target_rl,vel_target_lr_rev)
ydata = np.append(counts_rl_cal,counts_lr_cal_rev)

#Normalize the y-values
ydat = ydata/max(ydata)

#Convert data to dataframe
df = pd.DataFrame(list(zip(xdat, ydat)), columns =['xdat', 'ydat'])
df['E'] = E(df['xdat']) #Create df for x-data in terms of energy
df['npass'] = n_pass #Create df for number of passes
df['time'] = time_cal #Create df for calibrated time

#Calculate uncertainties
df['delta_v'] = ((25.1*df['npass'])/(df['time']/100)**2)/100 #Uncertainties in x-values
dcounts_lr = np.sqrt( (np.sqrt(counts_lr)/(time_lr/100))**2 + ((counts_lr/100)/((time_lr/100))**2)**2) #Uncertainty in lr counts
dcounts_rl = np.sqrt( (np.sqrt(counts_rl)/(time_rl/100))**2 + ((counts_rl/100)/((time_rl/100))**2)**2) #Uncertainty in rl counts
df['delta_counts']  = np.append(dcounts_rl, dcounts_lr[::-1]) #Append counts and reverse the order of lr so it shows with appropriate data
df['delta_counts'] = df['delta_counts']/max(ydata) #Normalize the y-data uncertainties

#Sort the values in order of increasing x values
dfs = df.sort_values('xdat')
#Remove this data point since it is almost a repeat and creates a bias which affects the fit.
dfs = df.drop([44])

#--------------------
#The section below can be used to plot on top of each other to show isomer shift
#fig, ax1 = plt.subplots(figsize=[30,20])
#ax1.errorbar(-vel_target_rl,counts_rl_cal/max(ydata),yerr=(dcounts_rl/max(ydata)),ls='', label='Measurements', marker='X',lw=4, zorder=1, ms=10, color='blue')
#ax1.grid()
#ax1.errorbar(vel_target_lr,counts_lr_cal/max(ydata),yerr=(dcounts_lr/max(ydata)),ls='', label='Measurements', marker='X',lw=4, zorder=1, ms=10, color='red')
#ax1.set_xlabel('Velocity [mm/s]', fontsize = 40)
#ax1.set_ylabel('# [Normalized Counts/s]', fontsize = 40)
#ax1.tick_params(axis='x', labelsize= 40)
#ax1.tick_params(axis='y', labelsize= 40)
#ax1.legend(fontsize = 40)
#plt.show()
#plt.close()
#-----------------------
#We can use the code below to create and plot individual fits to specific peaks ordered from left to right (Plot on top of each other not fully working)

#1st peak
#popt_n3, pcov_n3 = curve_fit(lorentz_func, dfs.xdat[0:11], dfs.ydat[0:11], bounds= ([-200,0,-6,0],[0,5,-4,5])) #Plot Normally
#popt_n3, pcov_n3 = curve_fit(lorentz_func, vel_target_lr[19:26], dfs.ydat[19:26], bounds= ([0,5,4,5,], [6,0,200,0])) #Plot on top of each other
#plt.plot(np.linspace(-6.5,-4,200),lorentz_func(np.linspace(-6.5,-4,200),*popt_n3)) #Plot Normally
#plt.plot(np.linspace(4,6.5,200),lorentz_func(np.linspace(4,6.5,200),*popt_n3), color='r') #Plot on top of each other

#2nd peak
#popt_n2, pcov_n2 = curve_fit(lorentz_func, vel_target_lr[12:18], dfs.ydat[12:18], bounds= ([-200,0,-4,-5],[0,5,-2,5])) #Plot Normally
#popt_n2, pcov_n2 = curve_fit(lorentz_func, vel_target_lr[12:18], dfs.ydat[12:18], bounds= ([2,5,0,5], [4,5,200,0])) #Plot on top of each other
#plt.plot(np.linspace(-4,-2,200),lorentz_func(np.linspace(-4,-2,200),*popt_n2)) #Plot Normally
#plt.plot(np.linspace(2,4,200),lorentz_func(np.linspace(2,4,200),*popt_n2), color='r') #Plot on top of each other

#3rd peak
#popt_n1, pcov_n1 = curve_fit(lorentz_func, dfs.xdat[19:26], dfs.ydat[19:26], bounds= ([-200,0,-2,-5],[0,5,0,5])) #Plot Normally
#popt_n1, pcov_n1 = curve_fit(lorentz_func, vel_target_lr[0:11], dfs.ydat[0:11], bounds= ([4,5,0,5], [6,0,200,0])) #Plot on top of each other
#plt.plot(np.linspace(-2.2,0,200),lorentz_func(np.linspace(-2.2,0,200),*popt_n1)) #Plot Normally
#plt.plot(np.linspace(0,2.2,200),lorentz_func(np.linspace(0,2.2,200),*popt_n1), color='r') #Plot on top of each other

#4th peak
#popt_p1, pcov_p1 = curve_fit(lorentz_func, dfs.xdat[26:33], dfs.ydat[26:33], bounds= ([-200,0,0,-5],[0,5,2.2,5]))
#plt.plot(np.linspace(0,2,200),lorentz_func(np.linspace(0,2,200),*popt_p1)) #Plot Normally
#plt.plot(np.linspace(0,2,200),lorentz_func(np.linspace(0,2,200),*popt_p1), color='b') #Plot on top of each other

#5th peak
#popt_p2, pcov_p2 = curve_fit(lorentz_func, dfs.xdat[33:41], dfs.ydat[33:41], bounds= ([-200,0,2.2,-5],[0,5,4.3,5]))
#plt.plot(np.linspace(2,4.3,200),lorentz_func(np.linspace(2,4.3,200),*popt_p2)) #Plot Normally
#plt.plot(np.linspace(2,4.3,200),lorentz_func(np.linspace(2,4.3,200),*popt_p2), color='b') #Plot on top of each other

#6th peak
#popt_p3, pcov_p3 = curve_fit(lorentz_func, dfs.xdat[41:52], dfs.ydat[41:52], bounds= ([-200,0,4.3,-5],[0,5,6.8,5]))
#plt.plot(np.linspace(4.3,6.8,200),lorentz_func(np.linspace(4.3,6.8,200),*popt_p3)) #Plot Normally
#plt.plot(np.linspace(4.3,6.8,200),lorentz_func(np.linspace(4.3,6.8,200),*popt_p3), color='b') #Plot on top of each other

plt.show()
plt.close()
#We use the code below to create a combine fit to the peaks
lower = [-200,-38,-30,-30,-40,-40,0,0,0,0,0,0,-6,-4,-2,0,2,4.4,-500]
upper = [0,0,0,0,0,0,5,5,5,5,5,5,-4,-2,0,2,4.2,6.8,500]
popt_comb, pcov_comb = curve_fit(lorentz_func_comb, dfs.xdat, dfs.ydat,bounds= (lower,upper), maxfev = 50000)


#We now find the fit parameters
pars = unumpy.uarray(popt_comb, np.sqrt(np.diag(pcov_comb)))
E_mean = np.mean(E(pars[6:12]))
E_widths = E(pars[6:12])
E_centrals = E(pars[12:18])
depths = pars[0:6]

#Plot the data along with the fit
fig, ax1 = plt.subplots(figsize=[40,20])
ax1.errorbar(dfs.xdat,dfs.ydat,yerr=dfs['delta_counts'],ls='', label='Measurements', marker='X',lw=4, zorder=1, ms=10)
ax1.grid()
ax2 = ax1.twiny()
ax2.set_xlabel('Energy [neV]', fontsize = 40)
ax1.set_xlabel('Velocity [mm/s]', fontsize = 40)
ax1.set_ylabel('# [Normalized Counts/s]', fontsize = 40)
ax2.scatter(dfs.E,dfs.ydat)
ax1.tick_params(axis='x', labelsize= 40)
ax1.tick_params(axis='y', labelsize= 40)
ax2.tick_params(axis='x', labelsize= 40)
ax1.plot(np.linspace(min(dfs.xdat),max(dfs.xdat),1000), lorentz_func_comb(np.linspace(min(dfs.xdat),max(dfs.xdat),1000), *popt_comb), color='red', lw= 7, alpha= 0.6,label = 'Least squares fit',zorder=0)
ax1.legend(fontsize = 40)
plt.show()
plt.close()
#---------------------------------------
#Calculate energy differences
delE_12 = E_centrals[5] - E_centrals[4]
delE_23 = E_centrals[4] - E_centrals[3]
delE_45 = E_centrals[2] - E_centrals[1]
delE_56 = E_centrals[1] - E_centrals[0]
delE_24 = E_centrals[4] - E_centrals[2]
delE_35 = E_centrals[3] - E_centrals[1]
delE_34 = E_centrals[3] - E_centrals[2]
delE_25 = E_centrals[4] - E_centrals[1]
delE_16 = E_centrals[5] - E_centrals[0]

#Calculate the Lande factors
g_12 = - delE_12/(3.15e-9*ufloat(33.3,1)*10e9)
g_23 = - delE_23/(3.15e-9*ufloat(33.3,1)*10e9)
g_45 = - delE_45/(3.15e-9*ufloat(33.3,1)*10e9)
g_56 = - delE_56/(3.15e-9*ufloat(33.3,1)*10e9)
g_24 = - delE_24/(3.15e-9*ufloat(33.3,1)*10e9)
g_35 = - delE_35/(3.15e-9*ufloat(33.3,1)*10e9)

#Calculate the average energies for the excited and ground states
E_exc = np.mean(np.array([delE_12, delE_23, delE_45, delE_56]))
E_grd = -np.mean(np.array([delE_24, delE_35]))

#Calculate the average Lande factor for the ground and excited states
g_exc = -E_exc/(3.15e-9*ufloat(33.3,1)*10e9)
g_grd = -E_grd/(3.15e-9*ufloat(33.3,1)*10e9)

#Calculate the isomeric shift
E_iso = np.mean(np.array([delE_16, delE_25, delE_34])) #E_iso = np.mean(E_centrals)
