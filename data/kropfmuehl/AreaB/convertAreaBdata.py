import numpy as np
from saem.maredata import Mare2dEMData
from saem import CSEMData
# %%
marefile = "AreaB_Bz.emdata"
mare = Mare2dEMData(marefile)
print(mare)
# %%
TX = []
for i in range(2):
    x, y = np.loadtxt("Tx{:d}.dat".format(i+1), unpack=True)
    TX.append(np.column_stack((x, y, x*0)))

mare.txpos = TX
mare.showPositions()
# %%
mare.saveData()  # all transmitters
for i in range(len(TX)):  # into separate transmitters
    mare.saveData(tx=i+1, topo=1)

raise SystemExit
# The next part is for individual data manipulation (filtering etc.),
# however only for single Tx (unless CSEMSurvey class is ready).
# %% Now re-read the individual files into the CSEMData class
for i in range(len(TX)):
    self = CSEMData(marefile.replace(".emdata", "_B_Tx{:d}.npz".format(i+1)))
    self.radius = 100
    self.line[:] = 1
    self.cmp = [0, 0, 1]
    # self.filter(6.)
    # self.filter(18.)
    # self.filter(30.)
    # self.filter(47.)
    self.showData()
    self.generateDataPDF(amphi=False, log=True, alim=[3, 3], figsize=(7,9))
# %%
sdfsdfs
# %%
mare.saveData(tx=1, topo=1)
mare.saveData(tx=2, topo=1)
mare.saveData(topo=1)
# %%
self = CSEMData(marefile.replace(".emdata", "_B_Tx4.npz"))
print(self)
self.cmp = [1, 0, 1]
# self.generateDataPDF(figsize=(9, 4), amphi=0)
# dsfsfs
# self.showData(0, amphi=0, log=1, alim=[-3, 3])
# %% every second frequency
self.filter(fInd=np.hstack((0, np.arange(2, self.nF))))
# self.filter(fInd=np.arange(3, self.nF-1))
self.showLineData(amphi=False, log=True, alim=[-3, 3])
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
self.line[:] = 1
# self.detectLines()
# %%
self.showPos()
self.showData(0, amphi=0, log=1, alim=[-3, 3])
# self.basename = "schleiz-L6-9take2Tx2"
self.basename = "schleiz-L6-_16freq_Tx2"
self.saveData(cmp=[1, 0, 1])
if 0:
    self.generateDataPDF(figsize=(9, 4), amphi=0)  # , what="error")
# %%
# new = CSEMData(self.basename+"BxBz.npz")
# new.showPos()
