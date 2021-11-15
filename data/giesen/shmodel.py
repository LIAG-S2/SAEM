# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:21:17 2017

@author: Rochlitz.R
"""

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg


# %% define survey parameters
invmod = 'GiesenLine7RotBx'
invmesh = 'Prisms'
inv_dir = "inv_results/" + invmod + "_" + invmesh

pgmesh = pg.load('meshes/mesh_create/' + invmesh + '.bms')
# %% save results
sigma = np.load(inv_dir + '/inv_model.npy')
res = 1. / sigma
# %% plot inv model
fig, ax = plt.subplots(figsize=(14, 8))
ax2, cbar = pg.show(pgmesh, res, ax=ax, cMap="Spectral", colorBar=True,
                    logScale=True, cMin=2, cMax=200,
                    xlabel='x [m]', ylabel='z [m]',
                    label=r'$\rho$ [$\Omega$m]', pad=0.8)

# cbar.ax.set_xlabel(r'$\sigma$ [S/m]', labelpad=4)
ax.figure.savefig(invmod+"-result2.pdf")
# %%
with np.load(invmod+".npz", allow_pickle=True) as ALL:
    freqs = ALL["freqs"]
    tx = ALL["tx"]
    print(tx)
    DATA = ALL["DATA"]
    rxs = [data["rx"] for data in DATA]
    tx_ids = [data["tx_ids"] for data in DATA]
    cmps = [data["cmp"] for data in DATA]

# %%
nF = len(freqs)
nC = len(cmps[0])
nR = rxs[0].shape[0]
respR, respI = np.split(np.load(invmod+"-response.npy"), 2)
resp = np.reshape(respR+respI*1j, [nF, nR, nC])
# %%
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
pa = ax[0].matshow(np.log10(np.abs(resp[:, :, 0])), cMap="Spectral_r")
pa.set_clim(-2, 0)
pp = ax[1].matshow(np.angle(resp[:, :, 0])*180/np.pi, cMap="hsv")
pp.set_clim(90, 180)
ax[0].set_ylim(ax[0].get_ylim()[::-1])
