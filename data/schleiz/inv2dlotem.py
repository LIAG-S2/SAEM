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


def topo_f(x, y=None):
    return x*0.
    # return(x/20. + np.sin(x*1e-2) * 10.)


# %% define mesh paramters
invmod = 'Ball_B_Tx1'
#invmod = 'Ball_B_Tx12456'
invmesh = 'Prisms'

dataR, dataI = [], []
errorR, errorI = [], []
with np.load(invmod+".npz", allow_pickle=True) as ALL:
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
sig_bg = 1e-2
rx_tri = mu.refine_rx(rxs[0], 1., 30.)  # needs to be in loop!
rx_tri = []
for rxi in rxs:
    rx_tri.extend(mu.refine_rx(rxi, 1., 30.))
# rx_tri = [mu.refine_rx(rxi, 1., 30.) for rxi in rxs]

bound = 200
minrx = min([min(data["rx"][:, 0]) for data in DATA])
maxrx = max([max(data["rx"][:, 0]) for data in DATA])
##############################################################################
# %% generate 2.5D prism inversion mesh
P = PrismWorld(name=invmesh,
               x_extent=[minrx-bound, maxrx+bound],
               x_reduction=300.,
               y_depth=1000.,
               z_depth=1000.,
               n_prisms=80,
               tx=[txi for txi in tx],
               surface_rx=rx_tri,
               prism_area=50000,
               prism_quality=34,
               x_dim=[-1e5, 1e5],
               y_dim=[-1e5, 1e5],
               z_dim=[-1e5, 1e5],
               topo=topo_f,
               )

for rx in rxs:
    P.PrismWorld.add_rx(rx)

# %%

P.PrismWorld.call_tetgen(tet_param='-pDq1.3aA', print_infos=False)
pgmesh = pg.load('meshes/mesh_create/' + invmesh + '.bms')
# pgmesh = P.xzmesh  # is 3D
if 0:
    pg.show(pgmesh)
    sdfsfsdf

# %% run inversion
datavec = np.hstack((dataR, dataI))
errorvec = np.abs(np.hstack((errorR, errorI))/datavec)

fop = MultiFWD(invmod, invmesh, pgmesh, list(freqs), cmps, tx_ids,
               skip_domains, sig_bg, n_cores=140, ini_data=datavec)
fop.setRegionProperties("*", limits=[1e-4, 1])
# set up inv
inv = pg.Inversion()
inv.setForwardOperator(fop)
C = pg.matrix.GeostatisticConstraintsMatrix(mesh=pgmesh, I=[50, 10])
fop.setConstraints(C)
dT = pg.trans.TransSymLog(1e-6)
inv.dataTrans = dT

# run inversion
invmodel = inv.run(datavec, errorvec, lam=40, zWeight=0.3,
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
                    logScale=True, cMin=2, cMax=1000,
                    xlabel='x [m]', ylabel='z [m]',
                    label=r'$\rho$ [$\Omega$m]', pad=0.8)

# cbar.ax.set_xlabel(r'$\sigma$ [S/m]', labelpad=4)
# ax.figure.savefig("out.pdf")
np.save(invmod+"-response.npy", inv.response)
fop.jacobian().save("jacobian.bmat")
