import numpy as np
import matplotlib.pyplot as plt

from saem.maredata import Mare2dEMData

self = Mare2dEMData("Ball.emdata")
print(self)
self.chooseF(every=3)
print(self)
for i in range(6):
    self.saveData(tx=i+1)

self.saveData()
# %%
if 0:
    # %%
    TX1B = self.getPart(tx=1, typ="B", clean=True)  # , clean=True)
    print(TX1B)
    TX1B.saveData()
    # %%
    mat = TX1B.getDataMatrix("Bx")
    plt.matshow(np.abs(mat))
    plt.matshow(np.angle(mat))
    # %%
    TX2B = self.getPart(tx=[1, 2], typ="B", clean=True)  # , clean=True)
    print(TX2B)
    # %%
    TX1E = self.getPart(tx=1, typ="E", clean=True)  # , clean=True)  # , typ=)
    print(TX1E)
