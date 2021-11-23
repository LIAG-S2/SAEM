# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:21:17 2017

@author: Rochlitz.R
"""

import numpy as np
import pygimli as pg
from custEM.meshgen.invmesh_tools import PrismWorld
from custEM.meshgen.meshgen_tools import BlankWorld
from custEM.meshgen import meshgen_utils as mu
from custEM.inv.inv_utils import MultiFWD


###############################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # MESH GENERATION # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
###############################################################################

# set numpy seed at the beginning to ensure same random numbers for noise
np.random.seed(999)


def topo_f(x, y=None):

    return x*0.
    # return(x/20. + np.sin(x*1e-2) * 10.)


# %% define mesh paramters
synthmesh = 'Synth'

tx = [np.array([[-50., -200., 0.],
                [-20., 200., 0.]]),
      np.array([[570., -200., 0.],
                [530., 200., 0.]])]

n_rx = 41
rx = np.zeros((n_rx, 3))
rx[:, 0] = np.linspace(50., 450., n_rx)
rx = mu.assign_topography(rx, topo=topo_f, z=40.)
rx_tri = mu.refine_rx(rx, 1., 30.)

rxs = [rx, rx]

# %% define survey parameters
invmod = 'Plate_Brick_double_Tx_9freqs'
synthmod = '100_10_1000_Ohm'

freqs = [30., 60., 100., 300., 600., 1000., 3000., 6000., 10000.]
cmp = [['H_x', 'H_z']] * 2
txs = [[0], [1]]
skip_domains = [0, 1]
sig_bg = 1e-2


# %% genreate a synthetic model
Synth = BlankWorld(name=synthmesh,
                   x_dim=[-2e3, 2e3],
                   y_dim=[-2e3, 2e3],
                   z_dim=[-2e3, 2e3],
                   topo=topo_f,
                   preserve_edges=True,
                   )

Synth.build_surface(insert_line_tx=tx)
Synth.build_halfspace_mesh()

Synth.add_plate(dx=200., dy=500., dz=30., origin=[100., 0.0, -100.0],
                dip=70., dip_azimuth=0., )
Synth.add_brick([300., -300., -200.], [450., 300., -50.])

[Synth.add_rx(rx) for rx in rxs]
Synth.add_paths(rx_tri)

# -D not required here as new algorithm works fine for such simple meshes
Synth.call_tetgen(tet_param='-pq1.3aA', print_infos=False)


# %% calculate synthetic model and apply noise to data
F = MultiFWD(synthmod, synthmesh, 99, freqs=freqs, cmp=cmp, txs=txs,
             sig_bg=sig_bg, n_cores=140, p_fwd=2, skip_domains=[0, 1])
# 99=dummy für pgmesh
# skip_domains,
# better use three conductivities instead of skip_domains and sig_bg
# synthmesh (name) should also be a vtk mesh to be

# calucalte synthetic data
synthData = F.response([1e-1, 1e-4])  # conductive plate and resistive brick
# F.import_fields()       # import existing fields if F.response is
# not called because synthetic data already exist

noise = 0.03        # in % / 100
n_level = 1e-7      # lowest detectable signal amplitude

abs_error = np.abs(synthData) * noise + n_level
rel_error = np.abs(abs_error/synthData)
noisydata = synthData + np.random.randn(len(synthData)) * abs_error
# REAL

dataR, dataI = np.split(noisydata, 2)
errorR, errorI = np.split(abs_error, 2)
# %%
nF = len(freqs)
nP = len(rxs)
iD = 0
DATA = []
for iP in range(nP):
    nT = len(txs[iP])
    nR = len(rxs[iP])
    nC = len(cmp[iP])
    nData = nT*nR*nC*nF
    dataRp = dataR[iD:iD+nData].reshape([nT, nF, nR, nC])
    dataIp = dataI[iD:iD+nData].reshape([nT, nF, nR, nC])
    errorRp = errorR[iD:iD+nData].reshape([nT, nF, nR, nC])
    errorIp = errorI[iD:iD+nData].reshape([nT, nF, nR, nC])
    iD += nData

    data = dict(dataR=dataRp, dataI=dataIp, errorR=errorRp, errorI=errorIp,
                tx_ids=txs[iP], rx=rxs[iP], cmp=cmp[iP])
    DATA.append(data)
# %% save them to NPY
np.savez("mydata.npz",
         tx=tx,
         freqs=freqs,
         DATA=DATA,
         origin=[0, 0, 0],  # global coordinates with altitude
         rotation=0
         )
