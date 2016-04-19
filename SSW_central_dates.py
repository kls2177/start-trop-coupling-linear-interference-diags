
#load packages
import numpy as np
from netCDF4 import Dataset
import cPickle as pickle

#user defined inputs
datdir = '/Users/Karen/Desktop/ERAI'
var = 'u'
fileend = '.allyrs.60N10hPa.nc'
month = ['09', '10', '11', '12', '01', '02', '03', '04', '05']

#open netcdf files
varnames = dict()
for i in range(len(month)):
    varout      = var + month[i]
    fname       = datdir + '/' + var + '.' + month[i] + fileend
    nc          = Dataset(fname)
    varnames[i] = nc.variables[var][:,:,0,0]

#create SSW seasons (Sept-May)
var_ssws = np.concatenate((varnames[0][0:varnames[0].shape[0]-1,:], varnames[1][0:varnames[0].shape[0]-1,:],
            varnames[2][0:varnames[0].shape[0]-1,:], varnames[3][0:varnames[0].shape[0]-1,:],
            varnames[4][1:varnames[0].shape[0],:], varnames[5][1:varnames[0].shape[0],:],
            varnames[6][1:varnames[0].shape[0],:],varnames[7][1:varnames[0].shape[0],:],
            varnames[8][1:varnames[0].shape[0],:]),axis=1)

#create time variable for plotting
time = [x for x in range(var_ssws.shape[1])]

#####################################################
#Now, search for SSWs using the two standard criteria

#First criterion (u < 0)
ssw_date = []
ssw_year = []
for j in range(var_ssws.shape[0]):
    i = 0
    while (i < len(time)):
        if var_ssws[j,i] < 0:
            ssw_date.append(i)
            ssw_year.append(j)
        i = i + 1

#Second criterion - events must be separated by 20 CONSECUTIVE days of westerly winds

#EASTERLY_DATE_FILTER: Function that filters easterly days to extract SSW central dates
def easterly_date_filter(i,n,sep):
    if (len(time) - (ssw_date[i] + n)) >= 20:
        for j in range(n,20+n-1):
            u_tmp = var_ssws[ssw_year[i],ssw_date[i]+j]
            if u_tmp > 0:
                sep = sep + 1
        if sep == 20: #20 consecutive days of westerlies
            count = 0
            ssw_date_new.append(ssw_date[i])
            ssw_year_new.append(ssw_year[i])
            for r in range(ssw_date[i],ssw_date[i]+n):
                if var_ssws[ssw_year[i],r] > 0:
                    count = count + 1
            i = i + n - count #go to next SSW
            n = 1
            sep = 1
            return [i, n, sep]
        else:
            n = n + 1 #did not find 20 consecutive days of westerlies yet, repeat
            sep = 1
            return [i, n, sep]
    else:
        i = i + 1
        n = 1
        return [i, n, sep]

#Second criterion - apply function
i = 0
n = 1
sep = 1
ssw_date_new = []
ssw_year_new = []
sswlist = [i,n,sep]

while sswlist[0] < len(ssw_year):

    #Treat first easterly day differently (i.e., no reference to separation between this easterly day and the previous).
    if sswlist[0] == 0:
        sswlist = easterly_date_filter(sswlist[0],sswlist[1],sswlist[2])

    #this section filters easterly dates within the same year (ie, multiple SSWs per year)
    elif (sswlist[0] != 1) and (ssw_date[sswlist[0]-1]+20 < ssw_date[sswlist[0]]) and (ssw_year[sswlist[0]-1] == ssw_year[sswlist[0]]):
        sswlist = easterly_date_filter(sswlist[0],sswlist[1],sswlist[2])

    #this section filters easterly dates in subsequent years
    elif (sswlist[0] != 1) and (ssw_year[sswlist[0]-1] != ssw_year[sswlist[0]]):
        sswlist = easterly_date_filter(sswlist[0],sswlist[1],sswlist[2])

    else: #keep looping over the end of the time series (final warming) until the index for the next year arrives
        sswlist[0] = sswlist[0] + 1
        n = 1
        sep = 1

#combine dates and year into one array for pickling
ssw_central_dates = np.vstack((ssw_date_new,ssw_year_new))

#pickle the dates for later use
with open('ERAI_ssw_central_dates.pickle','wb') as fp:
    pickle.dump(ssw_central_dates,fp)
