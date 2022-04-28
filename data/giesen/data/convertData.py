# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:08:18 2021

@author: Ronczka.M
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.io as sio
import imageio
from saem import CSEMData
from matplotlib.backends.backend_pdf import PdfPages
import pyproj
import os


# %% convert Mat files

for tx in ['1a', '1b', '2']:
    if tx == '1a':
        path = '20211004/'
        txpos = np.array([[559497.46, 5784467.953],
                          [559026.532, 5784301.022]]).T
    if tx == '1b':
        path = '20220124/'
        txpos = np.array([[559497.46, 5784467.953],
                          [559026.532, 5784301.022]]).T
    elif tx == '2':
        path = '20220209/'
        txpos = np.array([[559650.46, 5784095.953], [559130.532, 5783925.022],
                          [559035.532, 5783870.022]]).T
    
    filename = path + 'data*.mat'
    testsite = 'Giesen'
    flight = 'all'
    
    # raise SystemExit
    self = CSEMData(datafile=filename, txPos=txpos, txalt=5)
    self.basename = 'tx' + tx + ''
    self.saveData()