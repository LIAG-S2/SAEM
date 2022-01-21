import numpy as np
from saem.maredata import Mare2dEMData
from saem import CSEMData
# %%
marefile = "L06_07_08_09_Tx1_Tx2_BxByBz_47deg_masked.emdata"
mare = Mare2dEMData(marefile)
print(mare)
# %%
if 1:
    tx1 = np.loadtxt("tx1.pos")
    tx2 = np.loadtxt("tx2.pos")
    tx1 = np.column_stack((tx1, np.zeros(len(tx1))))
    print(tx1)
    print(tx2)
    mare.txpos = [tx2[:, [1, 0, 2]], tx1[:, [1, 0, 2]]]
    mare.showPositions()
# %%
mare.saveData(tx=1, topo=1)
mare.saveData(tx=2, topo=1)
mare.saveData(topo=1)
# %%
self = CSEMData(marefile.replace(".emdata", "_B_Tx2.npz"))
print(self)
self.cmp = [1, 0, 1]
# self.generateDataPDF(figsize=(9, 4), amphi=0)
self.showData(0, amphi=0, log=1, alim=[-3, 3])
# %% every second frequency
self.filter(fInd=np.arange(0, self.nF, 2))
print(self.DATA.shape, self.DATAX.shape)
# %% remove data close to transmitter
print(self.DATA.shape, self.DATAX.shape)
dTx = np.abs(self.rx-np.mean(self.tx))
self.filter(nInd=(dTx > 200))
print(self.DATA.shape, self.DATAX.shape)
dTx = np.abs(self.rx-np.mean(self.tx))
self.filter(nInd=(dTx < 2500))
print(self.DATA.shape, self.DATAX.shape)
# %% every fourth receiver
self.filter(nInd=np.arange(0, self.nRx, 2))
print(self.DATA.shape, self.DATAX.shape)
self.detectLines()
# %%
self.showPos()
self.showData(0, amphi=0, log=1, alim=[-3, 3])
self.basename = "schleiz-L6-9take2Tx2"
self.saveData(cmp=[1, 0, 1])
if 0:
    self.generateDataPDF(figsize=(9, 4), amphi=0)  # , what="error")
# %%
# fig, ax = plt.subplots()
# ax.plot(self.rx, self.ry)