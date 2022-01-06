import numpy as np
# import matplotlib.pyplot as plt
from saem import CSEMData
import pyproj
import pygimli as pg


# %% import transmitter (better by kmlread)
txgeo = np.array([[7.921688738, 53.498075347],
                  [7.921194641, 53.492855984]]).T
utm = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
txPos = np.fliplr(utm(*txgeo))

self = CSEMData(datafile="newdata_f1.mat", txPos=txPos, txalt=5)
# self.filter(fmin=100, fmax=25000)
self.filter(fmin=50, fmax=25000)
self.filter(f=12000)
# self.filter(f=6000)
# self.filter(f=29000)
print(self.f)
print(self)
# self.ry[self.line==2] += 10
# self.basename += "shift"
# self.generateDataPDF(figsize=(9, 5))  # linewise=True)
# self.showData(nf=1)
self.createDepthVector(rho=30)  # , nl=12)
self.depth *= 0.7
self.model = np.ones_like(self.depth) * 100
# %%
self.setPos(30)
self.cmp = [1, 1, 1]
ikw = dict(absError=0.0015, relError=0.04, lam=20)
self.invertSounding(**ikw) #check=True)
dsfsfsd
# self.showSounding()
# %%
self.line[:9]=0
# self.line[3:8] = 0
self.removeNoneLineData()
# %%
lkw = dict(alim=[-3, 0], plim=[90, 180])
ax = self.showLineData(line=1, **lkw)
ax[0][0].figure.savefig("line1Bx.pdf", bbox_inches="tight")
ax = self.showLineData(line=2, **lkw)
ax[0][0].figure.savefig("line2Bx.pdf", bbox_inches="tight")
# self.showSounding(amphi=False)
# %%
sdfsfsdf
# %%
# self.depth = np.hstack((0, pg.utils.grange(10, 200, n=15, log=True)))
# self.depth = np.hstack((0, np.cumsum(10**np.linspace(0.5, 1.6, 15))))
self.filter(fmin=50, fmax=20000)
self.createDepthVector(rho=30)
print(self.depth)

# %%
ikw = dict(absError=0.002, relError=0.05, lam=10)
self.invertSounding(25, show=True, **ikw)
# %%
dfsdfsd
# %%
self.invertLine(line=1, **ikw)
# %%
self.showLineData(line=1)
ax = self.showSection(cMin=10, cMax=200, zMax=120, label="resistivity (Ohmm)")
ax.set_aspect(1.0)
ax.figure.savefig("result4.pdf", bbox_inches="tight")
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
