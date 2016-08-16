#calc_timescales.py
#This package calculates annular mode timescales using the method of Simpson et al. (2011) and Mudryk and Kushner (2011)
#Written by: Karen L. Smith
#Date: 08/12/2016

#load packages
import numpy as np
import cPickle as pickle
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',size=10,weight='bold') #set default font size and weight for plots#load packages

#=================================================================================================

#make Gaussian filter
def GaussFilt(sigma,window):
    """defines Gaussian filter for smoothing"""
    x = np.linspace(-window/2, window/2, window)
    gaussFilter = np.exp(-x**2/(2*sigma**2))
    gaussFilter = gaussFilter/np.sum(gaussFilter)  #normalize
    return gaussFilter

#define exponential function for curve fit
def exponential_func(x, a, b):
    return a*np.exp(-b*x)

#fit exponential function
def ACFexpfit(ACF_smooth,lag):
    """fits an exponential to the ACF and returns the timescale (inverse of b)"""
    tau = np.zeros((ACF_smooth.shape[0]))
    for m in range(ACF_smooth.shape[0]):
        tautmp = ACF_smooth[m,:]
        days = np.linspace(1,lag,lag)
        popt,pcov = curve_fit(exponential_func,days,tautmp)
        tau[m]=1/np.abs(popt[1]) 
    return tau

#=================================================================================================
#=================================================================================================

def AMtimescales(varnames,lag,sigma,window):
    """Calculates annular mode timescales

    Args:
        Annular mode index at a particular level as a function of year and day (dict of months)
        Lag for exponential fit (may need to varied slightly depending on hemisphere of pressure level) (int)
        Standard deviation (sigma) for the Gaussian filter (float)
        Length of Gaussian filter smoothing window in days (int)

    Returns:
        Annular mode timescale and 95% confidence intervals as a function on day of the year
        Plot of annular mode timescale with 95% confidence intervals as a function on day of the year (AMtimescale.eps)
    """

    #make a time-series longer than 365 days to accomodate Gaussian smoothing
    #Jan-Dec of years 1:end
    var_mnth2 = np.concatenate((varnames[0][1:varnames[0].shape[0],:],varnames[1][1:varnames[0].shape[0],:],
                                varnames[2][1:varnames[0].shape[0],:], varnames[3][1:varnames[0].shape[0],:],
                                varnames[4][1:varnames[0].shape[0],:],varnames[5][1:varnames[0].shape[0],:],
                                varnames[6][1:varnames[0].shape[0],:],varnames[7][1:varnames[0].shape[0],:],
                                varnames[8][1:varnames[0].shape[0],:],varnames[9][1:varnames[0].shape[0],:],
                                varnames[10][1:varnames[0].shape[0],:],varnames[11][1:varnames[0].shape[0],:]),
                               axis=1)

    #Apr-Dec of years 0:(end-1)
    var_mnth = np.concatenate((varnames[3][0:varnames[0].shape[0]-1,:],varnames[4][0:varnames[0].shape[0]-1,:],
                               varnames[5][0:varnames[0].shape[0]-1,:],varnames[6][0:varnames[0].shape[0]-1,:],
                               varnames[7][0:varnames[0].shape[0]-1,:],varnames[8][0:varnames[0].shape[0]-1,:],
                               varnames[9][0:varnames[0].shape[0]-1,:],varnames[10][0:varnames[0].shape[0]-1,:],
                               varnames[11][0:varnames[0].shape[0]-1,:]),axis=1)
    
    var_series = np.concatenate((var_mnth,var_mnth2),axis=1) 

    #mask missing values
    var_series = np.ma.masked_greater(var_series,1e20)

    #calculate standardized anomalies
    var_std = np.ma.std(var_series,axis=0)
    var_s = np.ma.MaskedArray.anom(var_series/var_std,axis=0)

    #Define dimensions
    timelength = var_s.shape[1]
    lengthyear = var_s.shape[0]

    #calculate ACF as a function of day and lag of standardized field
    num    = np.zeros((lengthyear,timelength,lag))
    denom1 = np.zeros((lengthyear,timelength,lag))
    denom2 = np.zeros((lengthyear,timelength,lag))
    for i in range (lag):
        num[:,:,i]    = var_s*np.roll(var_s,-i,axis=1)
        denom1[:,:,i] = var_s**2
        denom2[:,:,i] = np.roll(var_s,-i,axis=1)*np.roll(var_s,-i,axis=1)

    ACF = np.sum(num,axis=0)/np.sqrt(np.sum(denom1,axis=0)*np.sum(denom2,axis=0))
    ACF_all = num/np.sqrt((denom1)*(denom2))

    #Gaussian smoothing of ACF over 181-day window
    gaussFilter = GaussFilt(sigma,window)

    ACF_tmp = np.zeros((window))
    ACF_smooth = np.zeros((ACF.shape[0]-(window-1),lag))

    gaussFilter = GaussFilt(sigma,window)
    for i in range(lag):
        for j in range(int((window-1)/2.0),ACF.shape[0]-int((window-1)/2.0)):
            ACF_tmp = np.convolve(ACF[j-int((window-1)/2.0):j+int((window-1)/2.0),i],
                                 gaussFilter,'same') #sliding 181-day window for Gaussian smoothing
            ACF_smooth[j-int((window-1)/2.0),i] = ACF_tmp[int((window-1)/2.0)]

    #Gaussian smoothing of ACF over 181-day window for each year to get a measure of variability (for confidence intervals)
    ACF_all_tmp = np.zeros((window))
    ACF_all_smooth = np.zeros((lengthyear,ACF_all.shape[1]-(window-1),lag))

    for k in range(lengthyear):
        for i in range(lag):
            for j in range(int((window-1)/2.0),ACF_all.shape[1]-int((window-1)/2.0)):
                ACF_all_tmp = np.convolve(ACF_all[k,j-int((window-1)/2.0):j+int((window-1)/2.0),i],
                                 gaussFilter,'same') #sliding 181-day window for Gaussian smoothing
                ACF_all_smooth[k,j-int((window-1)/2.0),i] = ACF_all_tmp[int((window-1)/2.0)]

    #confidence intervals
    #calculate standard deviation of ACF_all_smooth
    ACF_std = np.std(ACF_all_smooth,axis=0)
    
    #calculate two-tailed 95% t-value for lengthyear-1 degrees of freedom 
    alpha = 0.05/2
    dof = lengthyear-1
    upt = stats.t.ppf(1-alpha,dof) #finds t-value given alpha and dof
    dwnt = stats.t.ppf(alpha,dof)
    CI_up = (ACF_std/np.sqrt(lengthyear))*upt 
    CI_dwn = (ACF_std/np.sqrt(lengthyear))*dwnt 

    #create upper and lower bounds on timescale
    ACF_smooth_up = ACF_smooth + CI_up
    ACF_smooth_dwn = ACF_smooth + CI_dwn

    #fit exponentials to the ACFs to obtain timescales
    tau = ACFexpfit(ACF_smooth,lag)
    tau_dwn = ACFexpfit(ACF_smooth_dwn,lag)
    tau_up = ACFexpfit(ACF_smooth_up,lag)

    #plot for first 365 days
    series = np.linspace(1,365,365)
    fig = plt.figure(figsize=(18,48))
    ax = plt.subplot2grid((6,12),(0,0),colspan=4)
    ax.fill_between(series,tau_dwn[0:365],tau_up[0:365],
                facecolor='lightgray',edgecolor='lightgray')
    ax.plot(series,tau[0:365],'-',color='k',linewidth=2)
    ax.set_xlim(0,366)
    x = [0,32,63,93,124,154,185,216,244,275,305,336,366]
    monthstr = ['J','A','S','O','N','D','J','F','M','A','M','J','J']
    ax.set_xticks(x)
    ax.set_xticklabels([monthstr[i] for i in range(len(x))])
    ax.set_xlabel('Months',fontweight='bold')
    ax.set_ylabel('Annular Mode Timescale (days)',fontweight='bold')
    ax.set_title('Annular Mode Timescale',fontweight='bold')

    #save figure
    plt.savefig('AMtimescales.eps',bbox_inches='tight')

    return tau,tau_dwn,tau_up
