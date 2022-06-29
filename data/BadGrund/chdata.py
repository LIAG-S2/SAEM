import numpy as np
from saem import CSEMData


# Variant 1: only first and last point
# txpos = np.array([[583000, 584500], [5742500, 5741300]])
# Variant 2: complete
txpos = np.genfromtxt("Tx2.pos").T[:, ::-1]
# print(np.cumsum(np.sqrt(np.sum(np.diff(txpos)**2, axis=0))))
# ground = 400  #  altitude of ground
# %% read in the old data and take every second data point (12->24m)
self = CSEMData(datafile="data/*.mat", txPos=txpos) # , txalt=ground)
self.showField("alt")
# %%
self.alt[:] = 70
self.simulate(rho=100, fullTx=True)
# %%
kw = dict(nf=15, amphi=True, alim=[1e-3, 1])
self.showData(**kw)
self.showData(what="response", **kw)
# %%
kw = dict(nf=15, amphi=False, alim=[1e-3, 1])
self.showData(**kw)
self.showData(what="response", **kw)
# %%
# sdfsdfs
# %%
self.generateDataPDF()
# self.generateDataPDF(amphi=True, alim=[1e-3, 1])
# self.generateDataPDF(amphi=False, alim=[1e-3, 1])
