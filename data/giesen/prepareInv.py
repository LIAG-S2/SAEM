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
self.line[414:421] = 0  # remove tx-near
self.filter(fmin=100, fmax=20000)
self.filter(f=12000)
self.filter(f=7000)
# %%
self.createDepthVector(rho=30)  # , nl=12)
self.depth *= 0.7
self.model = np.ones_like(self.depth) * 25
print(self.depth)
# %%
line = 7
npos = 440
self.setPos(npos)
self.showLineData(line, alim=[-3, 0])
# %%
self.rotatePositions(line=line)
self.showLineData(line, alim=[-3, 0])
# %%
self.cmp = [1, 0, 0]
self.showLineData(line, alim=[-2, 0], plim=[-90, 0])
# %%
self.saveData(line="all")
