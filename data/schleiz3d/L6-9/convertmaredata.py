import numpy as np
from saem.maredata import Mare2dEMData
from saem import CSEMData

# %%
marefile = "L06_07_08_09_Tx1_Tx2_BxByBz_47deg_masked.emdata"
if 0:
    mare = Mare2dEMData(marefile, flipxy=True)
    mare.DATA["Tx"] = 3 - mare.DATA["Tx"]
    if 1:
        tx1 = np.loadtxt("tx1.pos")
        tx2 = np.loadtxt("tx2.pos")
        mare.txpos = [tx1[:, [1, 0, 2]], tx2[:, [1, 0, 2]]]
        mare.showPositions()
else:
    mare = Mare2dEMData(marefile)
    mare.DATA["Tx"] = 3 - mare.DATA["Tx"]
    tx1 = np.loadtxt("tx1.pos")
    tx2 = np.loadtxt("tx2.pos")
    mare.txpos = [tx1, tx2]

mare.showPositions()
dsfsfsf
# %%
mare.saveData(tx=1, topo=1)
mare.saveData(tx=2, topo=1)
mare.saveData(topo=1)
# %% Tx1
self = CSEMData(marefile.replace(".emdata", "_B_Tx1.npz"))
self.cmp = [1, 0, 1]

self.detectLines()
self.line[262:353] = 1
self.line[169:262] = 2
self.line[86:169] = 3
self.line[0:86] = 4
self.line += 5
# self.showLineData(6)
# asd

self.filter(fInd=np.arange(3, self.nF-1))
dTx = np.abs(self.rx-np.mean(self.tx))
self.filter(nInd=(dTx > 320.))
self.basename = "schleiz-Tx1new2"
self.reduceNoisy(aErr=0.0001, rErr=1.)
self.saveData(cmp=[1, 0, 1], aErr=0.001, rErr=0.05)
# %%
self.basename = "schleiz-Tx1new2"
new = CSEMData(self.basename+"BxBz.npz")
# new.showField(self.line)
new.showLineFreq(line=6, nf=4, what='data')
new.showLineFreq(line=6, nf=4, what='relError', log=True, alim=[0.01, 10.])
# %% Tx2
self = CSEMData(marefile.replace(".emdata", "_B_Tx2.npz"))
self.cmp = [1, 0, 1]

self.detectLines()
self.line[124:256] = 1
self.line[380:511] = 2
self.line[256:380] = 3
self.line[0:124] = 4
self.line += 5

self.filter(fInd=np.arange(4, self.nF-1))
self.filter(f=48.)
self.filter(f=66.)
# self.showLineData(6)
dTx = np.abs(self.rx-np.mean(self.tx))
self.filter(nInd=(dTx > 320.))
self.basename = "schleiz-Tx2new2"
self.reduceNoisy(aErr=0.0001, rErr=1.)
self.saveData(cmp=[1, 0, 1], aErr=0.001, rErr=0.05)

# %%
self.basename = "schleiz-Tx2new2"
new = CSEMData(self.basename+"BxBz.npz")
# new.showField(self.line)
new.showLineFreq(line=6, nf=7, what='data')
new.showLineFreq(line=6, nf=7, what='relError', log=True, alim=[0.01, 10.])
