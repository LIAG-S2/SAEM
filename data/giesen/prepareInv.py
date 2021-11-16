import numpy as np
import matplotlib.pyplot as plt
from saem import CSEMData


# %% import transmitter (better by kmlread)
txgeo = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txgeo, txalt=70,
                basename="giesen")
self.DATAY *= -1  # why?
self.cmp = [1, 1, 1]
# %%
self.filter(fmin=100, fmax=20000)
self.filter(f=12000)
self.filter(f=7000)
# %%
self.createDepthVector(rho=30)  # , nl=12)
self.depth *= 0.7
# print(self.depth)
self.model = np.ones_like(self.depth) * 25
# %%
line = 7
# %%
self.rotatePositions(line=line)
# self.showLineData(line, alim=[-3, 0])
# %%
self.cmp = [1, 0, 1]
self.showLineData(line, alim=[-2, 0], plim=[-90, 0])
# %%
# %% compute distance to transmitter
self.line[self.rx < 100] = 0
self.line[self.rx > 700] = 0
self.showField("line")
# self.showPos()
self.setPos(0)
self.saveData(line="all")
self.filter(nInd=np.nonzero(self.line > 0)[0])
self.filter(nInd=np.nonzero(self.line < 15)[0])
self.showPos()
# %%
self.filter(nInd=np.arange(0, len(self.rx), 4))
self.basename += "take4"
self.saveData()
self.showPos()
