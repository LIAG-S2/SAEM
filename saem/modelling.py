import numpy as np
from empymod import bipole
import pygimli as pg


def fwd(res, dep, inp, freqs):
    """Call empymods function bipole with the above arguments."""
    assert len(res) == len(dep), str(len(res)) + "/" + str(len(dep))
    OUT = bipole(res=np.concatenate(([2e14], res)),
                 depth=dep, freqtime=freqs, **inp)

    my = 4e-7 * np.pi
    OUT *= my * 1e9

    return OUT


class fopSAEM(pg.Modelling):
    def __init__(self, depth, cfg, f, cmp=[0, 0, 1]):
        """Initialize the model."""
        super().__init__()
        self.dep = -np.abs(depth)  # RHS pointing up
        self.cfg = cfg
        self.cmp = cmp
        self.f = f
        self.mesh1d = pg.meshtools.createMesh1D(len(self.dep))
        self.setMesh(self.mesh1d)

    def response(self, model):
        """Forward response."""
        resp = []
        if self.cmp[0]:
            self.cfg['rec'][3:5] = (0, 0)  # x
            resp.extend(fwd(model, self.dep, self.cfg, self.f))
        if self.cmp[1]:
            self.cfg['rec'][3:5] = (90, 0)  # y
            resp.extend(fwd(model, self.dep, self.cfg, self.f))
        if self.cmp[2]:
            self.cfg['rec'][3:5] = (0, 90)  # z
            resp.extend(fwd(model, self.dep, self.cfg, self.f))

        return np.hstack((np.real(resp), np.imag(resp)))

    def createStartModel(self, data):
        return pg.Vector(len(self.dep), 100)
