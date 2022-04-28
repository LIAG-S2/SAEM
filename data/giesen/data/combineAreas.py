# import matplotlib as mpl
# mpl.use('pdf')

import numpy as np
from saem import CSEMSurvey, CSEMData, Mare2dEMData


# %% Tx 1 and 2 data sets
file1 = "tx1aBxByBz.npz"
file2 = "tx1bBxByBz.npz"
file3 = "tx2BxByBz.npz"

self = CSEMSurvey(file1)
self.addPatch(file2)
self.addPatch(file3)

xoff = 559000
yoff = 5784000

for pi in range(3):
    self.patches[pi].detectLinesAlongAxis('y')
    # if pi in [0, 1]:
    #     self.patches[pi].DATAY *= -1
    if pi == 0:
        self.patches[pi].filter(every=2)
    # self.patches[pi].filter(every=4)
    self.patches[pi].filter(fmin=90, fmax=6000)
    self.patches[pi].filter(minTxDist=100)
    self.patches[pi].tx -= xoff
    self.patches[pi].rx -= xoff
    self.patches[pi].ty -= yoff
    self.patches[pi].ry -= yoff
    # self.patches[pi].estimateError(ignoreErr=False, useMax=True,
    #                                absError=1e-3, relError=0.06)
    self.patches[pi].estimateError(absError=1e-3, relError=0.05)
    self.patches[pi].deactivateNoisyData(aErr=0.0001, rErr=0.5)
    self.patches[pi].basename = 'GiesenArea' + str(pi+1)
    self.patches[pi].saveData(cmp=[0, 0, 1])
    self.patches[pi].saveData(cmp=[0, 1, 0])
    self.patches[pi].saveData(cmp=[0, 1, 1])
self.basename='Giesen'
self.saveData(cmp=[0, 0, 1])
self.saveData(cmp=[0, 1, 0])
self.saveData(cmp=[0, 1, 1])

# # %% merge everything
# self = CSEMSurvey()
# self.addPatch('data/Tx1BxBz_every' + str(skip) + '.npz')
# self.addPatch('data/Tx2BxBz_every' + str(skip) + '.npz')
# self.addPatch('data/Tx11BxBz_every' + str(skip) + '.npz')
# self.addPatch('data/Tx13BxBz_every' + str(skip) + '.npz')

# self.basename='data/Combined_every' + str(skip) + ''
# self.origin = origin
# self.angle = angle
# self.cmp = cmp
# self.saveData()