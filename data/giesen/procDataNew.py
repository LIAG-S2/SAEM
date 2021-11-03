import numpy as np
import matplotlib.pyplot as plt
from saem import CSEMData


# %% import transmitter (better by kmlread)
txpos = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
self.DATAX *= -1  # why?
self.showField(self.line)
# %%
self.line[414:421] = 0
self.filter(fmin=100, fmax=20000)
self.filter(f=12000)
self.filter(f=7000)
# %%
self.setPos(440, show=True)  # middle
# %%
self.createDepthVector(rho=30)  # , nl=12)
self.depth *= 0.7
print(self.depth)
# %%
self.cmp[0] = 0  # no x (Tx)
self.cmp[1] = 1
self.cmp[2] = 0
ikw = dict(absError=0.0015, relError=0.04, lam=20)
self.invertSounding(**ikw)
dfdg
# %%
line = 7
self.invertLine(line=line, **ikw)
ax = self.showSection(cMin=3, cMax=200)
ax.figure.savefig(f"line{line}-resultY.pdf", bbox_inches="tight")
# %%
dfsfsdf
# %%
self.invertLine(**ikw)
self.generateModelPDF()
