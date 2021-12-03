import numpy as np
from saem.maredata import Mare2dEMData

self = Mare2dEMData("GOS_raw_inversion_ByBz.emdata")
print(self)
self.showPositions()
# %%
ax = self.showPositions(True, markersize=1, save=True)
# ax.figure.savefig(self.basename+"-pos.pdf", bbox_inches="tight", dpi=300)
# %%
asdadasd
# %%
self.chooseF(every=2)
print(self)
print(self.f)
self.basename += "f2"
self.generateDataPDF()
# %%
self.saveData(topo=1)
for tx in [1, 2, 3]:
    self.saveData(tx=tx, topo=1)
