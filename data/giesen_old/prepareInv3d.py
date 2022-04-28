import numpy as np
from saem import CSEMData


# %% import transmitter (better by kmlread)
txgeo = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txgeo, txalt=70,
                basename="giesen")
self.filter(fmin=50, fmax=20000)
self.filter(f=12000)
self.showData(0)
# %%
rot = CSEMData(datafile="data_f*.mat", txPos=txgeo, txalt=70,
               basename="giesen")
rot.rotatePositions()
rot.showPos()
self.rx -= rot.origin[0]
self.ry -= rot.origin[1]
self.tx -= rot.origin[0]
self.ty -= rot.origin[1]
# %%
n = np.nonzero((rot.rx > 100)*(rot.rx < 700))[0]
# %%
self.filter(nInd=n)
self.filter(nInd=np.arange(0, len(self.rx), 4))
self.removeNoneLineData()
self.showPos()
self.basename += "-take4org"
self.showData(0)
sdfssdfsf
# %%
self.saveData(cmp=[1, 1, 1])
for i in range(3):
    cmp = [0, 0, 0]
    cmp[i] = 1
    self.saveData(cmp=cmp)
