import numpy as np
from saem.maredata import Mare2dEMData

self = Mare2dEMData("L07_bybz_edited.emdata")
self.saveData(topo=1)
# %%
ax = self.showPositions(True)
ax.figure.savefig(self.basename+"-pos.pdf", bbox_inches="tight", dpi=300)
# %%
lotem = Mare2dEMData("Ball.emdata")
ax = lotem.showPositions(True)
ax.figure.savefig(self.basename+"-pos.pdf", bbox_inches="tight", dpi=300)
# self.saveData()
# %%
# for tx in range(5):
#     part1 = self.getPart(tx=tx+1)
#     print(np.unique(part1.DATA["Freq"]))
#
