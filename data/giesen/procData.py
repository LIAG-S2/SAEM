import numpy as np
import matplotlib.pyplot as plt
from saem import CSEMData
import pygimli as pg
# from pygimli.viewer.mpl import drawModel1D




# %% import transmitter (better by kmlread)
txpos = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
self = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
self.DATAX *= -1  # why?
self.filter(fmin=50, fmax=24000)
print(self)
# self.generateDataPDF()
# self.showData(nf=1)
# self.showField("alt", background="BKG")
self.cmp[0] = 1
self.cmp[1] = 1
self.invertSounding(nrx=20)
# resp=self.inv1d.response.array()
# %%

# ax = self.showSounding(response=resp)
# plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
# self.showSounding(nrx=20)
# self.showData(nf=1)
# self.generateDataPDF()

