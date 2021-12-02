import numpy as np
from saem.maredata import Mare2dEMData

self = Mare2dEMData("L07_bybz_edited.emdata")
self.showPositions()
self.saveData()
# %%
for tx in range(5):
    part1 = self.getPart(tx=tx+1)
    print(np.unique(part1.DATA["Freq"]))
