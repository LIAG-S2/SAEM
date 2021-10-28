import numpy as np
import matplotlib.pyplot as plt
from saem import CSEMData


# %% import transmitter (better by kmlread)
txpos = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
self.DATAX *= -1  # why?
self.showField(self.line)
self.filter(fmin=100, fmax=20000)
self.filter(f=12000)
self.filter(f=7000)
# self.filter(f=5000)
# self.filter(f=900)
print(self.f)
print(self)
# self.showData(nf=1)
# self.generateDataPDF()
# self.showField("alt", background="BKG")
# %%
self.setPos(320, show=True)  # middle
# self.setPos(333, show=True)  # close to transmitter
# self.setPos(310, show=True)
# self.cmp = [1, 1, 1]
# self.showSounding(amphi=False)
# %%
# self.depth = np.hstack((0, np.cumsum(10**np.linspace(0.8, 2, 15))))
print(self.depth)
# %%
self.cmp[0] = 0  # no x (Tx)
self.cmp[1] = 1
self.cmp[2] = 0
# %
ikw = dict(absError=0.001, relError=0.03, lam=3)
self.invertSounding(**ikw)
# %%
line = 6
self.invertLine(line=line, **ikw)
ax = self.showSection(cMin=3, cMax=200)
ax.figure.savefig(f"line{line}-resultY.pdf", bbox_inches="tight")
