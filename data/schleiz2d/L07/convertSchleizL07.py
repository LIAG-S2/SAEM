import numpy as np
from saem import CSEMData, CSEMSurvey, Mare2dEMData

marefile = "L07_Tx1_Tx2_ByBz_47deg_masked.emdata"
# %%
self = CSEMSurvey(marefile, flipxy=True)
print(self)
self.filter(maxTxDist=4000)
self.showPositions()[0].savefig("pos.pdf")
self.showData(line=1)
self.saveData()
# %%
sdfsf
mare = Mare2dEMData(marefile, flipxy=True)
print(mare)
mare.saveData()
for i in range(2):
    mare.saveData(tx=i+1)

kw = dict(amphi=False, log=True, alim=[-3, 3], figsize=(10, 5))
# %%
i = 1
self = CSEMData(mare.basename + "_B_Tx{:d}.npz".format(i))
for f in [7, 15, 22, 48, 67]:
    self.filter(f=f)

self.showLineData(**kw)
self.saveData("Tx1.npz")
# %%
i = 2
self = CSEMData(mare.basename + "_B_Tx{:d}.npz".format(i))
for f in [10]:  # 7, 15, 22, 48, 67]:
    self.filter(f=f)

self.saveData("Tx2.npz")
# %% self.showLineData()
self.showLineData(**kw)
# %%
self.simulate(1000, show=True, **kw)
self.showLineData(what="response", **kw)
self.showLineData(what="misfit", **kw)
# %%
for rho in [10, 100, 1000, 10000]:
    self.simulate(rho, show=True, name=str(rho), **kw)
