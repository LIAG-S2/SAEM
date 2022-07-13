import numpy as np
import matplotlib
# matplotlib.use("pdf")
import matplotlib.pyplot as plt
from pygimli.viewer.mpl import underlayBKGMap
from saem import CSEMData
from pyproj import Proj


txpos = np.array([[582000, 584000],
                  [5744000, 5742000]]).T
ground = 400  #  altitude of ground
# %% read in the old data and take every second data point (12->24m)
self = CSEMData(datafile="L*.mat", txPos=txpos, txalt=ground)
self.showField("alt")
self.generateDataPDF()
sdfsdfsdf
# self.filter(every=2)
# %%
# from hdf5storage import loadmat
from scipy.io import loadmat

matfile = "L01_Source_LIAG_Tx02_Ncyc16_Ltsregress.mat"
ALL = loadmat(matfile)

bla = loadmat(matfile)["ztfs"][0][0]
# %%
matfile = matfile.replace("L1", "L2")
bla2 = loadmat(matfile)["ztfs"][0][0]
# %%
for i in range(20):
    bla[i] = np.concatenate((bla[i], bla2[i]), axis=-1)
# %%
sdfsdfsdf
# %%
f = 1./bla[12]
# time = bla[16]
gx, gy, gz = bla[17]
utm = Proj(proj='utm', zone=32, ellps='WGS84')
ggx, ggy = utm(gx, gy)
z = bla[18][0]
y, x = bla[19]
DATA = np.squeeze(bla[13])
ERR = np.squeeze(bla[14])
DN = np.squeeze(bla[15])
# %%
fig, ax = plt.subplots()
ax.plot(x, y, ".-")  # , ggx, ggy, ".-")
ax.set_aspect(1.0)
xl = ax.get_xlim()
ax.set_xlim(xl[0]-2100, xl[1]+2100)
underlayBKGMap(ax, mode="DOP")  # , uuid='139409f9-5953-1cf5-f19c-75262cfa13d6')
#fig.savefig("L1.pdf", bbox_inches="tight", dip=300)
fig.savefig("L1.pdf", dpi=200)
# %%
fig, ax = plt.subplots()
ax.plot(y, gz, "x-", y, bla[18][0], "+-")
