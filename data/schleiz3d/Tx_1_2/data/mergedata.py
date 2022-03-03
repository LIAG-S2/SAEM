import numpy as np
from saem import CSEMSurvey, CSEMData, Mare2dEMData
from saem.plotting import underlayBackground
import matplotlib.pyplot as plt

# %%
marefile = "L06_07_08_09_Tx1_Tx2_BxByBz_47deg_masked.emdata"
mare = Mare2dEMData(marefile, flipxy=True)
mare.DATA["Tx"] = 3 - mare.DATA["Tx"]

if 1:
    tx1 = np.loadtxt("tx1.pos")
    tx2 = np.loadtxt("tx2.pos")
    mare.txpos = [tx1[:, [1, 0, 2]], tx2[:, [1, 0, 2]]]
    #mare.showPositions()

mare.showPositions(globalCoordinates=True)
plt.savefig('map.pdf')
asd

# %% save mare files
mare.saveData(tx=1, topo=1)
mare.saveData(tx=2, topo=1)
mare.saveData(topo=1)

# %% merge
self = CSEMSurvey()

tx1 = CSEMData(marefile.replace(".emdata", "_B_Tx1.npz"))
tx1.cmp = [1, 0, 1]
tx1.detectLines()
tx1.line[262:353] = 1
tx1.line[169:262] = 2
tx1.line[86:169] = 3
tx1.line[0:86] = 4
tx1.line = 10 - tx1.line

tx2 = CSEMData(marefile.replace(".emdata", "_B_Tx2.npz"))
tx2.cmp = [1, 0, 1]
tx2.detectLines()
tx2.line[124:256] = 1
tx2.line[380:511] = 2
tx2.line[256:380] = 3
tx2.line[0:124] = 4
tx2.line = 10 - tx2.line
self.addPatch(tx1)
self.addPatch(tx2)

# %% Tx1
self.patches[0].filter(fInd=np.arange(3, len(self.patches[0].f)-5))
#self.patches[0].filter(nInd=np.arange(86, 169))
self.patches[0].filter(minTxDist=320)
self.patches[0].deactivateNoisyData(aErr=0.0001, rErr=1.)
self.patches[0].showField(self.patches[0].line)
# %% Tx2
self.patches[1].filter(fInd=np.arange(3, len(self.patches[1].f)-5))
#self.patches[1].filter(nInd=np.arange(256, 380))
self.patches[1].filter(minTxDist=320)
self.patches[1].deactivateNoisyData(aErr=0.0001, rErr=1.)
self.patches[1].showField(self.patches[1].line)

# %% show pos
f, ax = self.showPositions()
underlayBackground(ax, "BKG")

# # %% save data
# self.estimateError(relError=0.05, absError=1.5e-3)
# self.showData(nf=8, what="data")
# self.origin = [701449.9, 5605079.4, 0.]
# self.angle = 47.
# self.basename = 'line07-tx12'
# self.saveData(cmp=[1, 0, 1])