import numpy as np
import matplotlib.pyplot as plt
from saem import CSEMData
import pygimli as pg
# from pygimli.viewer.mpl import drawModel1D




# %% import transmitter (better by kmlread)
txpos = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f2.mat", txPos=txpos, txalt=70)
self.DATAX *= -1  # why?
# self.filter(fmin=50, fmax=20000)
# self.filter(f=12000)
# self.filter(f=7000)
# self.filter(f=5000)
print(self.f)
print(self)
# self.showData(nf=1)
# %%
self.setPos(5)
self.cmp[0] = 0
self.cmp[1] = 1
self.showSounding()
# %%
sdffs
# self.showField("alt", background="BKG")
# self.generateDataPDF()
self.invertSounding(nrx=20)
# resp=self.inv1d.response.array()
# %%

# ax = self.showSounding(response=resp)
# plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
# self.showSounding(nrx=20)
# self.showData(nf=1)
# self.generateDataPDF()

