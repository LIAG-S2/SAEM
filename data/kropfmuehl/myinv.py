# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:21:17 2017

@author: Rochlitz.R
"""

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
from custEM.meshgen.invmesh_tools import PrismWorld
from custEM.meshgen import meshgen_utils as mu
from custEM.inv.inv_utils import MultiFWD

xt, zt = np.loadtxt("topo.txt", unpack=True)


def topo_f(x, y=None):
    return np.interp(x, xt, zt)

# %% define mesh paramters
dataname = 'P5f2_B_Tx12'
invmod = dataname + '_l40'
invmesh = 'Prisms'

dataR, dataI = [], []
errorR, errorI = [], []
with np.load(dataname+".npz", allow_pickle=True) as ALL:
    freqs = list(ALL["freqs"])
    tx = ALL["tx"]
    print(tx)
    DATA = ALL["DATA"]
    rxs = [data["rx"] for data in DATA]
    # tx_ids = [[int(txi) for txi in data["tx_ids"]] for data in DATA]
    tx_ids = [data["tx_ids"] for data in DATA]
    cmps = [data["cmp"] for data in DATA]
    for i, data in enumerate(DATA):
        dataR = np.concatenate([dataR, data["dataR"].ravel()])
        dataI = np.concatenate([dataI, data["dataI"].ravel()])
        errorR = np.concatenate([errorR, data["errorR"].ravel()])
        errorI = np.concatenate([errorI, data["errorI"].ravel()])

skip_domains = [0, 1]
sig_bg = 3e-3

refm_size = 1.
rxs_resolved = mu.resolve_rx_overlaps(rxs, refm_size)

rx_tri = mu.refine_rx(rxs_resolved, refm_size, 30.)

bound = 200
minrx = min([min(data["rx"][:, 0]) for data in DATA])
maxrx = max([max(data["rx"][:, 0]) for data in DATA])

##############################################################################
# %% generate 2.5D prism inversion mesh
P = PrismWorld(name=invmesh,
               x_extent=[minrx-bound, maxrx+bound],
               x_reduction=500.,
               y_depth=1000.,
               z_depth=1200.,
               n_prisms=200,
               tx=[txi for txi in tx],
               orthogonal_tx=[True] * len(tx),
               #surface_rx=rx_tri,
               prism_area=50000,
               prism_quality=34,
               x_dim=[-1e5, 1e5],
               y_dim=[-1e5, 1e5],
               z_dim=[-1e5, 1e5],
               topo=topo_f,
               )

P.PrismWorld.add_paths(rx_tri)
for rx in rxs:
    P.PrismWorld.add_rx(rx)

# %%

P.PrismWorld.call_tetgen(tet_param='-pDq1.3aA', print_infos=False)
pgmesh = pg.load('meshes/mesh_create/' + invmesh + '.bms')
# pgmesh = P.xzmesh  # is 3D
if 0:
    ax, cb = pg.show(pgmesh)
    for rx in rxs:
        ax.plot(rx[:, 0], rx[:, 2], ".")
    for txi in tx:
        for txii in txi:
            print(txii)
            ax.plot(txii[0], txii[2], "mv")


# %% run inversion
mask = np.isfinite(dataR+dataI+errorR+errorI)
datavec = np.hstack((dataR[mask], dataI[mask]))
errorvec = np.hstack((errorR[mask], errorI[mask]))
relerror = np.abs(errorvec/datavec)

fop = MultiFWD(invmod, invmesh, pgmesh, list(freqs), cmps, tx_ids,
               skip_domains, sig_bg, n_cores=140, ini_data=datavec,
               data_mask=mask)
fop.setRegionProperties("*", limits=[1e-4, 1])
# set up inv
inv = pg.Inversion(verbose=True)  # , debug=True)
inv.setForwardOperator(fop)
C = pg.matrix.GeostatisticConstraintsMatrix(mesh=pgmesh, I=[500, 80])
# fop.setConstraints(C)
dT = pg.trans.TransSymLog(1e-3)
inv.dataTrans = dT

# run inversion
invmodel = inv.run(datavec, relerror, lam=40,  # zWeight=0.3,
                   startModel=sig_bg, maxIter=10,
                   verbose=True, robustData=True)
# %% save results
np.save(fop.inv_dir + 'inv_model.npy', invmodel)
res = 1. / invmodel
pgmesh['sigma'] = invmodel  # np.load(fop.inv_dir + 'inv_model.npy')
pgmesh['res'] = res  # np.load(fop.inv_dir + 'inv_model.npy')
# pgmesh.setDimension(3)
# pgmesh.swapCoordinates(1, 2)
pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')
# %% plot inv model
fig, ax = plt.subplots(figsize=(14, 8))
ax2, cbar = pg.show(pgmesh, res, ax=ax, cMap="Spectral", colorBar=True,
                    logScale=True, cMin=5, cMax=5000,
                    xlabel='x [m]', ylabel='z [m]',
                    label=r'$\rho$ [$\Omega$m]', pad=0.8)

# cbar.ax.set_xlabel(r'$\sigma$ [S/m]', labelpad=4)
# ax.figure.savefig("out.pdf")
np.save(invmod+"-response.npy", inv.response)
fop.jacobian().save("jacobian.bmat")
