# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:21:17 2017

@author: Rochlitz.R
"""

import numpy as np
import pygimli as pg
from custEM.meshgen.invmesh_tools import PrismWorld
from custEM.meshgen import meshgen_utils as mu
from custEM.inv.inv_utils import MultiFWD


def topo_f(x, y=None):
    return x*0.
    # return(x/20. + np.sin(x*1e-2) * 10.)


# %% define mesh paramters
invmesh = 'Prisms'

dataR, dataI = [], []
errorR, errorI = [], []
with np.load("mydata.npz", allow_pickle=True) as ALL:
    freqs = ALL["freqs"]
    tx = ALL["tx"]
    print(tx)
    DATA = ALL["DATA"]
    rxs = [data["rx"] for data in DATA]
    tx_ids = [data["tx_ids"] for data in DATA]
    cmps = [data["cmp"] for data in DATA]
    for i, data in enumerate(DATA):
        dataR = np.concatenate([dataR, data["dataR"].ravel()])
        dataI = np.concatenate([dataI, data["dataI"].ravel()])
        errorR = np.concatenate([errorR, data["errorR"].ravel()])
        errorI = np.concatenate([errorI, data["errorI"].ravel()])

# rx = mu.assign_topography(rx, topo=topo_f, z=40.)
# rx_tri = mu.refine_rx(rx, 1., 30.)

# %% define survey parameters
invmod = 'Plate_Brick_double_Tx_9freqs'
synthmod = '100_10_1000_Ohm'

skip_domains = [0, 1]
sig_bg = 1e-2

##############################################################################
# %% generate 2.5D prism inversion mesh
P = PrismWorld(name=invmesh,
               x_extent=[-100., 600.],
               x_reduction=100.,
               y_depth=400.,
               z_depth=300.,
               n_prisms=80,
               tx=[txi for txi in tx],
               prism_area=5e2,
               prism_quality=31.2,
               x_dim=[-2e3, 2e3],
               y_dim=[-2e3, 2e3],
               z_dim=[-2e3, 2e3],
               topo=topo_f,
               )

pgmesh = pg.load('meshes/mesh_create/' + invmesh + '.bms')
# pgmesh = P.xzmesh  # is 3D
# pg.show(pgmesh)
# %%
for i, rx in enumerate(rxs):
    P.PrismWorld.add_rx(rx)

rx_tri = mu.refine_rx(rx, 1., 30.)  # needs to be in loop!
P.PrismWorld.add_paths(rx_tri)  #
# %%
# use -pD for good olf Constraint Delaunay tetrahedralization (CDT) instead of
# new boundary recovery algorithm (deafult -p option) in TetGen 1.6,
# The new algorithm freezes (at least it takes more than 10 min) if n_prisms
# becomes high. The -pD option makes definitely better meshes than TetGen 1.5.1
# with less nodes, but not as good as -p if that works.
P.PrismWorld.call_tetgen(tet_param='-pDq1.3aA', print_infos=False)


# %% Inversion parameters
n_iter = 11
# noise = 0.03        # in % / 100
# n_level = 1e-7      # lowest detectable signal amplitude

# %% run inversion
datavec = np.hstack((dataR, dataI))
errorvec = np.hstack((errorR, errorI))

fop = MultiFWD(invmod, invmesh, pgmesh, list(freqs), cmps, tx_ids, skip_domains,
               sig_bg, n_cores=140, ini_data=datavec)
fop.setRegionProperties("*", limits=[1e-4, 1])
# set up inv
inv = pg.Inversion()
inv.setForwardOperator(fop)
C = pg.matrix.GeostatisticConstraintsMatrix(mesh=pgmesh, I=[50, 10])
fop.setConstraints(C)
dT = pg.trans.TransSymLog(1e-4)
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
# pgmesh.swapCoordinates(1, 2)
# pgmesh.swapCoordinates(1, 2)
pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')

fig, ax = plt.subplots(figsize=(14, 8))
ax2, cbar = pg.show(pgmesh, res, ax=ax, cMap="Spectral", colorBar=True,
                    logScale=True, cMin=10, cMax=1000,
                    xlabel='x [m]', ylabel='z [m]',
                    label=r'$\rho$ [$\Omega$m]', pad=0.8)

# cbar.ax.set_xlabel(r'$\sigma$ [S/m]', labelpad=4)
# ax.figure.savefig(invmesh + invmodel + '.png')
od+"-response.npy", inv.response)
fop.jacobian().save("jac