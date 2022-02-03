import numpy as np
import matplotlib.pyplot as plt
from demclass import DEM
from saem.maredata import Mare2dEMData
from saem import CSEMData

mare = Mare2dEMData("L07_Tx1_Tx2_ByBz_47deg_masked.emdata", flipxy=False)
mare.angle = -mare.angle + 180
# %%
ax = mare.showPositions(True)
ax.plot(*mare.origin[:2], "w*", markersize=10)
# %%
dem = DEM("../Schleiz_dgm25_50x50km.asc")
# %%
ax = dem.show()
rxpos = mare.rxPositions()
ax.plot(rxpos[:, 0], rxpos[:, 1])
rxz = dem(rxpos[:, 0], rxpos[:, 1])
rxy, rxx = mare.rxPositions(False).T
xline = np.arange(-2500, 5500, 25)
si = np.argsort(rxx)
tz = np.interp(xline, rxx[si], rxz[si])
# %%
fig, ax = plt.subplots()
ax.plot(xline, tz, "b-", rxx, mare.rxpos[:, 2], "rx")
# ax.plot(rxx, rxz)
np.savetxt("topo.txt", np.column_stack((xline, tz)))
