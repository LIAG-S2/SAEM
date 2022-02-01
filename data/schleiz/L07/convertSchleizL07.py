# import numpy as np
# import matplotlib.pyplot as plt
from saem.maredata import Mare2dEMData
from saem import CSEMData

mare = Mare2dEMData("L07_Tx1_Tx2_ByBz_47deg_masked.emdata", flipxy=True)
# print(self)
# self.chooseF(every=3)
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
# %%
i = 2
self = CSEMData(mare.basename + "_B_Tx{:d}.npz".format(i))
for f in [10]:  # 7, 15, 22, 48, 67]:
    self.filter(f=f)
# %% self.showLineData()
self.showLineData(**kw)
# %%
self.simulate(1000, show=True, **kw)
self.showLineData(what="response", **kw)
self.showLineData(what="misfit", **kw)
# %%
for rho in [10, 100, 1000, 10000]:
    self.simulate(rho, show=True, name=str(rho), **kw)
