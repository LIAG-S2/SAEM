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
# %%
i = 1
self = CSEMData(mare.basename + "_B_Tx{:d}.npz".format(i))
for f in [7, 15, 22, 48, 67]:
    self.filter(f=f)

# self.showLineData()
self.showLineData(amphi=False, log=True, alim=[3, 3])
# %%
i = 2
self = CSEMData(mare.basename + "_B_Tx{:d}.npz".format(i))
for f in [10]:  # 7, 15, 22, 48, 67]:
    self.filter(f=f)

# self.showLineData()
self.showLineData(amphi=False, log=True, alim=[3, 3], figsize=(10, 5))
