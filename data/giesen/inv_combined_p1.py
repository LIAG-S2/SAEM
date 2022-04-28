#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rrochlitz
"""

from custEM.meshgen.meshgen_tools import BlankWorld
from custEM.meshgen import meshgen_utils as mu
from custEM.inv.inv_utils import MultiFWD
import numpy as np
import pygimli as pg
from saem import CSEMData

# ########################################################################### #
# # # # #        drone example 3D inversion of synthtic data          # # # # #
# ########################################################################### #

cmp = 'Bz'
invmod = 'p1_' + cmp + 'Area3'
invmesh = 'invmesh_' + invmod

aErr = 1e-3
dataname = 'data/GiesenArea3' + cmp
saemdata = np.load(dataname+".npz", allow_pickle=True)
print([freq for freq in saemdata["freqs"]])
sig_bg = 5e-3

# newid = [[0], [0], [1]]
# for ti, data in enumerate(saemdata['DATA']):
#     data.update({"tx_ids": newid[ti]})
#     print(data["tx_ids"])

# %% mesh generation

M = BlankWorld(name=invmesh,
                x_dim=[-1e4, 1e4],
                y_dim=[-1e4, 1e4],
                z_dim=[-1e4, 1e4],
                preserve_edges=True,
                inner_area_cell_size=1e3,
                outer_area_cell_size=1e6,
                )


invpoly = np.array([[-0.4e3, -0.7e3, 0.],
                    [0.8e3, -0.7e3, 0.],
                    [0.8e3, 1e3, 0.],
                    [-0.4e3, 1e3, 0.]])

txs = [mu.refine_path(saemdata['tx'][ti], length=25.) for ti in range(1)]

M.build_surface(insert_line_tx=txs)
M.add_inv_domains(-200., invpoly, x_frame=1e3, y_frame=1e3, z_frame=1e3,
                  cell_size=1e6)
M.build_halfspace_mesh()


allrx = mu.resolve_rx_overlaps([data["rx"] for data in saemdata["DATA"]], 1.)
rx_tri = mu.refine_rx(allrx, 1., 30.)
M.add_paths(rx_tri)

# add receiver locations to parameter file for all receiver patches
for rx in [data["rx"] for data in saemdata["DATA"]]:
    M.add_rx(rx)

M.call_tetgen(tet_param='-pq1.2aA', print_infos=False)

###############################################################################
# %% run inversion

# set up forward operator
fop = MultiFWD(invmod, invmesh, saem_data=saemdata, sig_bg=sig_bg,
                n_cores=60, p_fwd=1, start_iter=0)
fop.setRegionProperties("*", limits=[1e-4, 1])

# set up inversion operator
inv = pg.Inversion()
inv.setForwardOperator(fop)
inv.setPostStep(fop.analyze)
dT = pg.trans.TransSymLog(aErr)
inv.dataTrans = dT

# run inversion
invmodel = inv.run(fop.measured, fop.errors, lam=10., verbose=True,
                   startModel=fop.sig_0, maxIter=21)

###############################################################################
# %% post-processingelf
np.save(fop.inv_dir + 'inv_model.npy', invmodel)
res = 1. / invmodel
pgmesh = fop.mesh()
pgmesh['sigma'] = invmodel
pgmesh['res'] = res
pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')


#####
# fop = MultiFWD(invmod, invmesh, saem_data=saemdata, sig_bg=sig_bg,
#                n_cores=60, p_fwd=1, start_iter=0)
# fop.import_local_jacobian()
# print('done')