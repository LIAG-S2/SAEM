import numpy as np
# import matplotlib.pyplot as plt
from saem import CSEMData
import pyproj
# import pygimli as pg


# %% import transmitter (better by kmlread)
txgeo = np.array([[7.921688738, 53.498075347],
                  [7.921194641, 53.492855984]]).T
utm = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

self = CSEMData(datafile="data_f*.mat", txPos=utm(*txgeo), txalt=5)
self.DATAZ *= -1  # why?
self.filter(fmin=50, fmax=5000)
# self.filter(f=12000)
# self.filter(f=7000)
# self.filter(f=29000)
print(self.f)
print(self)
# self.showData(nf=1)
# %%
self.setPos(10)
self.cmp[0] = 1
self.cmp[1] = 0
self.showSounding(amphi=False)
# %%
self.invertLine(range(9, 32), maxIter=8)
# %%
ax = self.showSection(cMin=10, cMax=200, zMax=120, label="resistivity (Ohmm)")
ax.set_aspect(1.0)
ax.figure.savefig("result2.pdf", bbox_inches="tight")
# %%
# self.setPos(15)
# self.invertSounding(show=True)
# %%
# sdffs
# self.showField("alt", background="BKG")
# self.generateDataPDF()
# self.invertSounding(nrx=20)
# resp=self.inv1d.response.array()
# %%

# ax = self.showSounding(response=resp)
# plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
# self.showSounding(nrx=20)
# self.showData(nf=1)
# self.generateDataPDF()
