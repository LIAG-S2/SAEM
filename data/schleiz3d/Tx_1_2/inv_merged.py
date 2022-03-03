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

tx = '12'

invmod = 'Tx' + tx + 'p2_fine'
invmesh = 'schleiz_fine' + invmod

dataname = 'tx' + tx + 'BxBz'
saemdata = np.load('data/' + dataname+".npz", allow_pickle=True)
print([freq for freq in saemdata["freqs"]])

sig_bg = 2e-3

# DA = CSEMData('data/' + dataname+".npz")
# print(len(saemdata['freqs']), saemdata['freqs'])
# # DA.showData(amphi=False, nf=3)
# asd

# %% mesh generation

M = BlankWorld(name=invmesh,
               x_dim=[-6e3, 1e4],
               y_dim=[-4e3, 4e3],
               z_dim=[-1e4, 1e4],
               preserve_edges=True,
               t_dir='../',
               topo='Schleiz_dgm25_50x50km.asc',
               inner_area_cell_size=5e3,
               easting_shift=-saemdata['origin'][0],
               northing_shift=-saemdata['origin'][1],
               rotation=-float(saemdata['rotation']),
               outer_area_cell_size=1e6,
               )

inv_area1 = np.array([[-2e3, -0.8e3, 0.],
                      [5e3, -0.8e3, 0.],
                      [5e3, 0.8e3, 0.],
                      [-2e3, 0.8e3, 0.]])
inv_area2 = np.array([[-3e3, -1.6e3, 0.],
                      [6e3, -1.6e3, 0.],
                      [6e3, 1.6e3, 0.],
                      [-3e3, 1.6e3, 0.]])

txs = [mu.refine_path(saemdata['tx'][0], length=100.),
       mu.refine_path(saemdata['tx'][1], length=100.)]

M.build_surface(insert_line_tx=txs)
M.add_surface_anomaly(insert_paths=[inv_area1, inv_area2],
                      depths=[-1000., -1200.],
                      cell_sizes=[1e7, 1e10],
                      dips=[0., 0.],
                      dip_azimuths=[0., 0.],
                      marker_positions=[[0., 0., -0.1],
                                        [0., 1500., -0.1]])

M.build_halfspace_mesh()

# add receiver locations to parameter file for all receiver patches
allrx = mu.resolve_rx_overlaps([data["rx"] for data in saemdata["DATA"]], 5.)
rx_tri = mu.refine_rx(allrx, 5., 30.)
M.add_paths(rx_tri)
# add receiver locations to parameter file for all receiver patches
for rx in [data["rx"] for data in saemdata["DATA"]]:
    M.add_rx(rx)

M.overwrite_markers=[2, 3]
M.extend_world(6., 20., 5.)
M.call_tetgen(tet_param='-pq1.6aA', print_infos=False)

###############################################################################
# %% run inversion

# set up forward operator
fop = MultiFWD(invmod, invmesh, saem_data=saemdata, sig_bg=sig_bg,
               n_cores=72, p_fwd=2)
fop.setRegionProperties("*", limits=[1e-4, 1])

# set up inversion operator
inv = pg.Inversion()
inv.setForwardOperator(fop)
inv.setPostStep(fop.analyze)
dT = pg.trans.TransSymLog(1.5e-3)
inv.dataTrans = dT

# run inversion
invmodel = inv.run(fop.measured, fop.errors, lam=20., verbose=True,
                   startModel=fop.sig_0, maxIter=51)

###############################################################################
# %% post-processingelf
np.save(fop.inv_dir + 'inv_model.npy', invmodel)
res = 1. / invmodel
pgmesh = fop.mesh()
pgmesh['sigma'] = invmodel
pgmesh['res'] = res
pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')
#fop._jac.save(fop.inv_dir + 'jacobian')