# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:03:09 2022

@author: jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

bad_data = pd.read_table("C:/Users/jacob/Desktop/BCGS/Courses/Lab/Radio Astronomy/Radio_Lab_Data/sto25_N_A_spec_324.csv", skiprows=(6), delimiter=' ')
calib = pd.read_table("C:/Users/jacob/Desktop/BCGS/Courses/Lab/Radio Astronomy/Radio_Lab_Data/sto25_N_A_spec_320.csv", skiprows=(6), delimiter=' ')
calib.columns=['Chn_Num', 'Freq', '3','4','5','6','T_A']
data = pd.read_table("C:/Users/jacob/Desktop/BCGS/Courses/Lab/Radio Astronomy/Radio_Lab_Data/sto25_N_A_spec_325.csv", skiprows=(6), delimiter=' ')
data.columns=['Chn_Num', 'Freq', '3','4','5','6','T_A']
off_target = pd.read_table("C:/Users/jacob/Desktop/BCGS/Courses/Lab/Radio Astronomy/Radio_Lab_Data/sto25_N_A_spec_326.csv", skiprows=(6), delimiter=' ')
off_target.columns=['Chn_Num', 'Freq', '3','4','5','6','T_A']

def fitfunction(chnnum, T_A):
    fit = np.polyfit(chnnum,T_A ,5)
    p = np.poly1d(fit)
    a = fit[0]
    b = fit[1]
    c = fit[2]
    d = fit[3]
    e = fit[4]
    f = fit[5]
    x = np.linspace(1190, 1961, num=772)
    fit_equation = a*(x)**5 + b*(x)**4 + c*(x)**3 + d*(x)**2 + e*x + f
    return fit, fit_equation

def fitfunction2(chnnum, T_A, number):
    fit = np.polyfit(chnnum,T_A ,5)
    p = np.poly1d(fit)
    a = fit[0]
    b = fit[1]
    c = fit[2]
    d = fit[3]
    e = fit[4]
    f = fit[5]
    x = np.linspace(1000, 2500, num=number)
    fit_equation = a*(x)**5 + b*(x)**4 + c*(x)**3 + d*(x)**2 + e*x + f
    return fit, fit_equation

###################################################################################

plt.plot(calib.Chn_Num,calib.T_A,'k',label='S7 Spectrum') #Plot Initial Data
#Plot the data
#plt.title("Measured Calibration Source Spectrum")
#plt.xlabel('Channel Number')
#plt.ylabel('Antenna Temperature (K)')
#plt.legend()
#plt.margins(x=0)
#plt.show()

#Fist Step: hiding bandpass edges \n and HI 21-cm line emission
#Remove at bandpass edges and source
nosource1 = calib.loc[5:282]
nosource2 = calib.loc[428:750]
nosource = pd.concat([nosource1, nosource2])
#nosource = calib

#Create the polynomial fit
fit = np.polyfit(nosource.Chn_Num,nosource.T_A,5)
p = np.poly1d(fit)

#Plot the data
plt.title("Measured Calibration Source Spectrum \n and Baseline Fit")
plt.xlabel('Channel Number')
plt.ylabel('Antenna Temperature (K)')
plt.margins(x=0)


#Second Step: Subtract baseline from measurement
a = fit[0]
b = fit[1]
c = fit[2]
d = fit[3]
e = fit[4]
f = fit[5]
x = np.linspace(1190, 1961, num=772)
fit_equation = a*(x)**5 + b*(x)**4 + c*(x)**3 + d*(x)**2 + e*x + f
plt.plot(calib.Chn_Num,fit_equation,'-r', label='Baseline Fit', linewidth=1) #Plot Baseline Fit (1st plot)
plt.legend()
plt.show()
plt.close()

#Convert frequency to velocity:
c = 299792.458
freq_ratio = (calib.Freq/1.42041e9)
vel = (1-freq_ratio)*c

baseline =  fit_equation
subtracted_base = calib.T_A - fit_equation
fig, ax1 = plt.subplots()
ax1.plot(vel,subtracted_base,'k-', label='S7 Spectrum')
ax2 = ax1.twiny()
ax2.plot(calib.Freq/(1e6),subtracted_base, 'k-', alpha=0)
ax2.invert_xaxis()
plt.title("Calibration Source Spectrum")
ax2.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Antenna Temperature (K)')
ax1.set_xlabel('Velocity (km/s)')
ax1.legend()
ax1.margins(x=0)
ax2.margins(x=0)
plt.show()
plt.close()

print("Max Antenna Temp: " + str(subtracted_base.max()))

#Antenna Temp Uncertainty
rms = np.sqrt((np.sum(subtracted_base**2)/(len(subtracted_base))))
#rms2 = np.sqrt(np.mean(subtracted_base**2))
print("RMS T_A: " + str(rms))
#print("RMS2: " + str(rms2))

#Determine Conversion Factor
T_B = 96.3
T_A = np.max(subtracted_base)
alpha = T_B /T_A
print("Alpha: " + str(alpha))

#Uncertainty in Alpha
a_unc = (T_B/(T_A**2))*rms
print("Uncertainty in Alpha: " + str(a_unc))

#Calculate calibrated fit and T_B
fit, calib_FitEqn = fitfunction(nosource.Chn_Num,nosource.T_A*alpha)
subtracted_base_calibrated = calib.T_A*alpha - calib_FitEqn

#Brightness temperature uncertainty
rms2 = np.sqrt((np.sum(subtracted_base_calibrated**2)/(len(subtracted_base_calibrated))))
#rms2 = np.sqrt(np.mean(subtracted_base_calibrated**2))
print("RMS T_B: " + str(rms2))

#System Temperature
systemp = rms2*np.sqrt(6103.52*29.70)  #6103.52
print("T_sys: " + str(systemp))

fig, ax1 = plt.subplots()
ax1.plot(vel,subtracted_base_calibrated,'k-', label='S7 Spectrum')
ax2 = ax1.twiny()
ax2.plot(calib.Freq/(1e6),subtracted_base_calibrated, 'k-', alpha=0)
ax2.invert_xaxis()
plt.title("Corrected Calibration Source Spectrum")
ax2.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Brightness Temperature (K)')
ax1.set_xlabel('Velocity (km/s)')
ax1.legend()
ax1.margins(x=0)
ax2.margins(x=0)
plt.show()
plt.close()

###################################################################################


#Plot original data
#plt.plot(data.Chn_Num,data.T_A,'k',label='S7 Spectrum')


#For actual data
nosource_1 = data.loc[1400:1500]
nosource_2 = data.loc[1501:1900]
#nosource_1 = data.loc[2400:2500] #Actual values of where source is supposed to be
#nosource_2 = data.loc[2501:2600]
nosource_dat = pd.concat([nosource_1, nosource_2])

datazoom = data.loc[1400:1900]
offzoom = off_target.loc[1400:1900]
#datazoom = data.loc[2400:2600]
#offzoom = off_target.loc[2400:2600]


plt.plot(nosource_dat.Chn_Num, nosource_dat.T_A, 'k-', label="NGC1073 Spectrum")
#Plot the data
plt.title("Measured Source Spectrum \n and Baseline Fit")
plt.xlabel('Channel Number')
plt.ylabel('Antenna Temperature (K)')
plt.margins(x=0)

fit2, data_FitEqn = fitfunction2(nosource_dat.Chn_Num,nosource_dat.T_A, 1501)
#plt.plot(datazoom.Chn_Num,data_FitEqn,'-r', label='Baseline Fit', linewidth=1)
plt.plot(datazoom.Chn_Num,offzoom.T_A,'-r', label='Off-Target Baseline', linewidth=1)
plt.legend()
plt.show()
plt.close()

#plt.plot(offzoom.Chn_Num, offzoom.T_A)
#plt.plot(off_target.Chn_Num, off_target.T_A)
#plt.plot(data.Chn_Num,data.T_A,'k',label='S7 Spectrum')
#plt.show()
#plt.close()
#fit3, data_FitEqn2 = fitfunction2(nosource_dat.Chn_Num,nosource_dat.T_A, 16375)

#Subtracting the baseline starts here

#Convert frequency to velocity:
freq_ratio2 = (datazoom.Freq/1.42041e9)
vel2 = (1-freq_ratio2)*c

baseline2 =  off_target
subtracted_base_dat = (datazoom.T_A - offzoom.T_A)
subtracted_base_dat = np.clip(subtracted_base_dat, a_min=0, a_max=None)
fig, ax1 = plt.subplots()
ax1.plot(vel2,subtracted_base_dat,'k-', label='NGC1073 Spectrum')
ax2 = ax1.twiny()
ax2.plot(datazoom.Freq/(1e6),subtracted_base_dat, 'k-', alpha=0)
ax2.invert_xaxis()
#ax2.vlines(1420.405, -1, 2, colors='g',linestyles='dashed', label='21-cm Line')
ax1.axvline(x=0, c='g',linestyle='--', label='Expected Position')
plt.title("Source Spectrum")
ax2.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Antenna Temperature (K)')
ax1.set_xlabel('Velocity (km/s)')
ax1.legend()
#ax2.legend()
ax1.margins(x=0)
ax2.margins(x=0)
plt.show()
plt.close()

#Find maximum brightness temperature
T_bmax = subtracted_base_dat.max()*alpha
print('Max Brightness Temp: ' + str(T_bmax))
#Find uncertainty in maximum brightness temperature
#rms3 = np.sqrt(np.mean(subtracted_base_dat**2))
#print("RMS T_B: " + str(rms3))


subtracted_base_datcal = subtracted_base_dat*alpha #datazoom.T_A*alpha - offzoom.T_A*alpha 
rms3 = np.sqrt(np.mean(subtracted_base_datcal**2))
print("RMS T_B: " + str(rms3))

fig, ax1 = plt.subplots()
plot1 = ax1.plot(vel2,subtracted_base_datcal,'k-', label='NGC1073 Spectrum')
ax2 = ax1.twiny()
plot2 = ax2.plot(datazoom.Freq/(1e6),subtracted_base_datcal, 'k-', alpha=0)
ax2.invert_xaxis()
ax1.axvline(x=0, c='g',linestyle='--', label='Expected Position')
#ax1.axvline(x=61, c='g',linestyle='--', label='Expected Position')
plt.title("Calibrated Source Spectrum")
ax2.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Brightness Temperature (K)')
ax1.set_xlabel('Velocity (km/s)')
ax1.legend()
#.legend()
#lns = plot1 + plot2
#labels = [l.get_label() for l in lns]
#plt.legend(lns, labels, loc=0)
ax1.margins(x=0)
ax2.margins(x=0)
plt.show()
plt.close()

#integral = np.trapz(subtracted_base_datcal[1564:1612], x=vel2[1564:1612])
#subtracted_base_datcal = subtracted_base_datcal.reset_index()
integral = np.sum(subtracted_base_datcal[164:212]/0.09)
length = len((subtracted_base_datcal[164:212]))
uncert = np.sqrt(np.sum((10**2)*length)) #/(len(subtracted_base_datcal)))
print(integral)
print(uncert)
