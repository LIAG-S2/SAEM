import numpy as np
from saem import CSEMData


# %% import transmitter (better by kmlread)
txgeo = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txgeo, txalt=70,
                basename="giesen")
# self.DATAY *= -1  # why?
# %%
self.filter(fmin=50, fmax=20000)
self.filter(f=12000)
# self.filter(f=7000)
# self.saveData(cmp=[1,1,1])
# sdfsdfsfd
# self.showLineData(line, alim=[-2, 0], plim=[-90, 0])
# %%
self.showPos()
self.cmp = [1, 1, 0]
line = 7
self.showLineData(line, alim=[-1, -1], amphi=False, cmp=[0, 1, 0])
self.showLineData(line, alim=[-2, 0])#, plim=[-90, 90])
self.rotatePositions(ang=np.deg2rad(12))  # line=line)
self.showLineData(line, alim=[-2, 0])#, plim=[-90, 90])
self.showPos()


self.showLineData(line, alim=[-2, 0], plim=[-90, 0], cmp=[1,0,0], amphi=False)


self.showLineData(line, alim=[-1, -1], cmp=[1,0,0], amphi=False)

# %% compute distance to transmitter
self.line[self.rx < 100] = 0
self.line[self.rx > 700] = 0
self.removeNoneLineData()
# %%
self.basename += "-switchXY"
self.saveData(line=line, cmp=[0, 1, 0])
fsdfsfsd
self.saveData(line=line, cmp=[0, 0, 1])
self.saveData(line=line, cmp=[1, 0, 1])
sdfsfsf
self.generateDataPDF()
# self.generateDataPDF(linewise=True)
self.generateDataPDF(linewise=True, amphi=False, alim=[1., 1.], log=True)  # , log=True, alim=[1, 1])
self.saveData(line="all")
# self.filter(nInd=np.nonzero(self.line > 0)[0])
self.filter(nInd=np.nonzero(self.line < 15)[0])
# self.showPos()
# %% 3D inversion
self.filter(nInd=np.arange(0, len(self.rx), 4))
self.showPos()
self.basename += "take4"
# %%
self.saveData(cmp=[1, 0, 1])
# self.generateDataPDF(alim=[-3, 0], plim=[-90, 0])
self.saveData(cmp=[1, 0, 0])
self.saveData(cmp=[0, 0, 1])
self.saveData(cmp=[1, 1, 1])
