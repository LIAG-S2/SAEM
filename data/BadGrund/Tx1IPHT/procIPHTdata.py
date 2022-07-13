import numpy as np
from saem import CSEMData


txpos = np.genfromtxt("../Tx2.pos").T[:, ::-1]
# %% read in the old data and take every second data point (12->24m)
self = CSEMData(datafile="data/*.mat", txPos=txpos) # , txalt=ground)
dx = np.sqrt(np.diff(self.rx)**2+np.diff(self.ry)**2)
print(np.median(dx))
self.filter(every=5)
self.radius = 60
self.showField("line")
# %%
self.filter(fmin=12, fmax=1100)
self.filter(fInd=np.arange(0, len(self.f), 2))
# self.filter(f=10.)
# %%
self.showLineData(line=5)
# %%
self.rotate()
# %%
self.filter(minTxDist=300, maxTxDist=3000)
self.deactivateNoisyData(rErr=0.5)
self.estimateError()
self.deactivateNoisyData(rErr=0.5, aErr=0.01)
# %%
self.showData(line=5)
# %%
txdir = -1
self.cmp = [1, 0, 1]
self.saveData("Tx2IPHT_BxBz", txdir=txdir)
self.filter(every=2)
self.saveData("Tx2IPHT_BxBzE2", txdir=txdir)
self.filter(every=2)
self.saveData("Tx2IPHT_BxBzE4", txdir=txdir)
self.saveData("Tx2IPHT_BzE4", cmp=[0, 0, 1], txdir=txdir)
self.saveData("Tx2IPHT_BxE4", cmp=[1, 0, 0], txdir=txdir)
