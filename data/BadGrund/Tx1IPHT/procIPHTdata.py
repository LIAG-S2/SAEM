import numpy as np
from saem import CSEMData


txpos = np.genfromtxt("../Tx2.pos").T[:, ::-1]
self = CSEMData(datafile="data/*.mat", txPos=txpos)
dx = np.sqrt(np.diff(self.rx)**2+np.diff(self.ry)**2)
print("Median Rx distance: ", np.median(dx))
self.filter(every=5)
self.radius = 60  # make plot similar to WWU bird
# self.showField("line")
# %%
self.filter(fmin=12, fmax=1100)
self.filter(fInd=np.arange(0, len(self.f), 2))  # every second
# %%
self.showLineData(line=5)
# %%
self.filter(minTxDist=500, maxTxDist=3000)
self.deactivateNoisyData(rErr=0.5)
self.estimateError()  # 5%+1pV/A
self.deactivateNoisyData(rErr=0.5)
# %%
self.showData(line=5)
# %%
self.setOrigin([580000., 5740000.])
self.showField("line")
# %%
self.basename = "Tx2IPHT"
self.saveData()
self.saveData(cmp=[0, 1, 1])
self.filter(every=2)
self.basename += "_E2"
self.saveData()
self.filter(every=2)
self.basename = self.basename.replace("E2", "E4")
self.saveData()
# %%
self.saveData(cmp=[1, 0, 0])
self.saveData(cmp=[0, 1, 0])
self.saveData(cmp=[0, 0, 1])
