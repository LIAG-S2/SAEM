import numpy as np
from saem import CSEMData


# %% import transmitter (better by kmlread)
txgeo = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txgeo, txalt=70,
                basename="giesen")
self.DATAY *= -1  # why?
self.cmp = [1, 1, 1]
# %%
self.filter(fmin=50, fmax=20000)
self.filter(f=12000)
# self.filter(f=7000)
# %%
self.rotatePositions(line=7)
# %%
self.cmp = [1, 0, 1]
# self.showLineData(line, alim=[-2, 0], plim=[-90, 0])
# %%
# %% compute distance to transmitter
self.line[self.rx < 100] = 0
self.line[self.rx > 700] = 0
self.removeNoneLineData()
self.generateDataPDF()
# self.generateDataPDF(linewise=True)
self.generateDataPDF(linewise=True, amphi=False, alim=[1., 1.], log=True)  # , log=True, alim=[1, 1])
sdfsfsf
self.saveData(line="all")
# self.filter(nInd=np.nonzero(self.line > 0)[0])
self.filter(nInd=np.nonzero(self.line < 15)[0])
# self.showPos()
# %% 3D inversion
self.filter(nInd=np.arange(0, len(self.rx), 4))
self.basename += "take4"
self.cmp = [1, 0, 1]
self.saveData()
self.generateDataPDF(alim=[-3, 0], plim=[-90, 0])
self.cmp = [1, 0, 0]
self.saveData()
self.showPos()
