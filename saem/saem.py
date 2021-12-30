from glob import glob
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyproj

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D
from pygimli.viewer.mpl import showStitchedModels
from pygimli.core.math import symlog

from .plotting import plotSymbols, showSounding
from .modelling import fopSAEM, bipole


class CSEMData():
    """Class for CSEM frequency sounding."""

    def __init__(self, **kwargs):
        """Initialize CSEM data class

        Parameters
        ----------
        datafile : str
            data file to load
        basename : str [datafile without extension]
            name for data (exporting, figures etc.)
        txPos : array
            transmitter position as polygone
        rx/ry/rz : iterable
            receiver positions
        f : iterable
            frequencies
        alt : float
            flight altitude
        """
        self.basename = "noname"
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = [0, 0, 1]  # active components
        self.txAlt = kwargs.pop("txalt", 0.0)
        self.tx, self.ty = kwargs.pop("txPos", (None, None))
        self.rx = kwargs.pop("rx", np.array([100.0]))
        self.ry = np.zeros_like(self.rx)
        self.alt = kwargs.pop("alt", 0.0)
        self.rz = np.zeros_like(self.rx) * self.alt
        self.depth = None
        self.prim = None
        self.origin = [0, 0, 0]
        self.angle = 0
        self.A = np.array([[1, 0], [0, 1]])
        if "datafile" in kwargs:
            self.loadData(kwargs["datafile"])

        self.basename = kwargs.pop("basename", self.basename)

    def __repr__(self):
        """String representation of the class."""
        sdata = "CSEM data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))
        txlen = np.sqrt(np.diff(self.tx)**2+np.diff(self.ty)**2)[0]
        stx = "Transmitter length {:.0f}m".format(txlen)
        spos = "Sounding pos at " + (3*"{:1f},").format(*self.cfg["rec"][:3])

        return "\n".join((sdata, stx, spos))

    def loadData(self, filename):
        """Load data from mat file (WWU Muenster processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = glob(filename)
        assert len(filenames) > 0
        filename = filenames[0]
        MAT = loadmat(filename)
        if len(filenames) > 1:
            print("read "+filename)
        for filename in filenames[1:]:
            print("reading "+filename)
            MAT1 = loadmat(filename)
            assert len(MAT["f"]) == len(MAT1["f"]), filename+" nf not matching"
            assert np.allclose(MAT["f"], MAT1["f"]), filename+" f not matching"
            for key in MAT1.keys():
                if key[0] != "_" and key != "f":
                    MAT[key] = np.hstack([MAT[key], MAT1[key]])

        # self.f = np.squeeze=MAT["f"]*1.0
        self.rx, self.ry = self.utm(MAT["lon"][0], MAT["lat"][0])
        self.f = np.squeeze(MAT["f"]) * 1.0
        self.DATAX = MAT["ampx"] * np.exp(MAT["phix"]*np.pi/180*1j)
        self.DATAY = MAT["ampy"] * np.exp(MAT["phiy"]*np.pi/180*1j)
        self.DATAZ = MAT["ampz"] * np.exp(MAT["phiz"]*np.pi/180*1j)
        self.rz = MAT["alt"][0]
        self.alt = self.rz - self.txAlt
        self.createConfig()
        self.detectLines()

    def simulate(self, rho, thk):
        """Simulate data by assuming 1D layered model."""
        pass

    def rotateBack(self):
        """Rotate coordinate system back."""
        self.tx, self.ty = self.A.T.dot(np.array([self.tx, self.ty]))
        self.rx, self.ry = self.A.T.dot(np.vstack([self.rx, self.ry]))
        self.tx += self.origin[0]
        self.ty += self.origin[1]
        self.rx += self.origin[0]
        self.ry += self.origin[1]
        self.origin = [0, 0, 0]
        self.A = np.array([[1, 0], [0, 1]])
        for i in range(len(self.f)):
            # self.A.T.dot
            Bxy = self.A.dot(np.vstack((self.DATAX[i, :], self.DATAY[i, :])))
            self.DATAX[i, :] = Bxy[0, :]
            self.DATAY[i, :] = Bxy[1, :]

    def rotatePositions(self, ang=None, line=None):
        """Rotate positions so that transmitter is x-oriented."""
        self.rotateBack()  # always go back to original system
        self.origin = [np.mean(self.tx), np.mean(self.ty)]
        if ang is None:
            if line is None:
                ang = np.arctan2(np.diff(self.ty),
                                 np.diff(self.tx))[0] + np.pi / 2
            else:
                rx = self.rx[self.line == line]
                ry = self.ry[self.line == line]
                ang = np.median(np.arctan2(ry-ry[0], rx-rx[0]))

        self.A = np.array([[np.cos(ang), np.sin(ang)],
                           [-np.sin(ang), np.cos(ang)]])
        self.tx, self.ty = self.A.dot(np.array([self.tx-self.origin[0],
                                                self.ty-self.origin[1]]))
        self.tx = np.round(self.tx*10+0.001) / 10
        self.rx, self.ry = self.A.dot(np.vstack([self.rx-self.origin[0],
                                                 self.ry-self.origin[1]]))

        for i in range(len(self.f)):
            # self.A.dot
            Bxy = self.A.T.dot(np.vstack((self.DATAX[i, :], self.DATAY[i, :])))
            self.DATAX[i, :] = Bxy[0, :]
            self.DATAY[i, :] = Bxy[1, :]

        self.createConfig()  # make sure rotated Tx is in cfg
        self.angle = ang

    def detectLines(self, show=False):
        """Split data in lines for line-wise processing."""
        dx = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        sdx = np.hstack((0, np.diff(np.sign(np.diff(self.rx))), 0))
        sdy = np.hstack((0, np.diff(np.sign(np.diff(self.ry))), 0))
        self.line = np.zeros_like(self.rx, dtype=int)
        nLine = 1
        act = True
        for i in range(len(sdx)):
            if sdx[i] != 0:
                act = not act
                if act:
                    nLine += 1
            if sdy[i] != 0:
                act = not act
                if act:
                    nLine += 1
            if i > 0 and dx[i-1] > 50:
                act = True
                nLine += 1

            if act:
                self.line[i] = nLine

        if show:
            self.showField(self.line)

    def removeNoneLineData(self):
        """Remove data not belonging to a specific line."""
        self.filter(nInd=np.nonzero(self.line)[0])

    def filter(self, fmin=0, fmax=1e6, f=-1, nInd=None, nMin=None, nMax=None):
        """Filter data according to frequency range and indices."""
        bind = (self.f > fmin) & (self.f < fmax)  # &(self.f!=f)
        if f > 0:
            bind[np.argmin(np.abs(self.f - f))] = False

        ind = np.nonzero(bind)[0]
        self.f = self.f[ind]
        self.DATAX = self.DATAX[ind, :]
        self.DATAY = self.DATAY[ind, :]
        self.DATAZ = self.DATAZ[ind, :]
        if self.prim is not None:
            for i in range(3):
                self.prim[i] = self.prim[i][ind, :]

        if nMin is not None or nMax is not None:
            if nMin is None:
                nMin = 0
            if nMax is None:
                nMax = len(self.rx)

            nInd = range(nMin, nMax)

        if nInd is not None:
            for tok in ['alt', 'rx', 'ry', 'rz', 'line']:
                setattr(self, tok, getattr(self, tok)[nInd])

            self.DATAX = self.DATAX[:, nInd]
            self.DATAY = self.DATAY[:, nInd]
            self.DATAZ = self.DATAZ[:, nInd]
            if hasattr(self, 'MODELS'):
                self.MODELS = self.MODELS[nInd, :]
            if self.prim is not None:
                for i in range(3):
                    self.prim[i] = self.prim[i][:, nInd]

    def mask(self):
        pass

    def createConfig(self):
        """Create EMPYMOD input argument configuration."""
        self.cfg = {'src':
                    [self.tx[0], self.tx[1], self.ty[0], self.ty[1], 0.1, 0.1],
                    'rec': [self.rx[0], self.ry[0], -self.alt[0], 0, 90],
                    'strength': 1, 'mrec': True,
                    'srcpts': 5,
                    'htarg': {'pts_per_dec': 0, 'dlf': 'key_51_2012'},
                    'verb': 1}

    def setPos(self, nrx=0, position=None, show=False):
        """The ."""
        if position:
            dr = (self.rx - position[0])**2 + (self.ry - position[1])**2
            if self.verbose:
                print("distance is ", np.sqrt(dr))

            nrx = np.argmin(dr)

        self.cfg["rec"][:3] = self.rx[nrx], self.ry[nrx], -self.alt[nrx]
        self.dataX = self.DATAX[:, nrx]
        self.dataY = self.DATAY[:, nrx]
        self.dataZ = self.DATAZ[:, nrx]
        self.nrx = nrx
        if show:
            self.showPos()

    def showPos(self, ax=None, line=None):
        """Show positions."""
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.rx, self.ry, "b.", markersize=2)
        ax.plot(self.tx, self.ty, "r-", markersize=4)
        if hasattr(self, "nrx"):
            ax.plot(self.rx[self.nrx], self.ry[self.nrx], "bo", markersize=5)

        if line is not None:
            ax.plot(self.rx[self.line == line],
                    self.ry[self.line == line], "g-")

        ax.set_aspect(1.0)
        ax.grid(True)

    def skinDepths(self, rho=30):
        """Compute skin depth based on a medium resistivity."""
        return np.sqrt(rho/self.f) * 500

    def createDepthVector(self, rho=30, nl=15):
        """Create depth vector."""
        sd = self.skinDepths(rho=rho)
        self.depth = -np.hstack((0, pg.utils.grange(min(sd)*0.3, max(sd)*1.2,
                                                    n=nl, log=True)))
        # depth = -np.hstack((0, np.cumsum(10**np.linspace(0.8, 1.5, 15))))
        # return depth

    def invertSounding(self, nrx=None, show=True, check=False, depth=None,
                       relError=0.03, absError=0.001, **kwargs):
        """Invert a single sounding."""
        if nrx is not None:
            self.setPos(nrx)

        if depth is not None:
            self.depth = depth
        if self.depth is None:
            self.depth = self.createDepthVector()

        self.fop1d = fopSAEM(self.depth, self.cfg, self.f, self.cmp)
        self.fop1d.modelTrans.setLowerBound(1.0)
        if not hasattr(self, "model"):
            self.model = np.ones_like(self.depth) * 100

        if check:
            self.response1d = self.fop1d(self.model)
        else:
            data = []
            for i, cmp in enumerate(["X", "Y", "Z"]):
                if self.cmp[i]:
                    data.extend(getattr(self, "data"+cmp))

            self.inv1d = pg.Inversion(fop=self.fop1d)
            transModel = pg.trans.TransLogLU(1, 1000)
            self.inv1d.transModel = transModel
            datavec = np.hstack((np.real(data), np.imag(data)))
            absoluteError = np.abs(datavec) * relError + absError
            relativeError = np.abs(absoluteError/datavec)
            self.model = self.inv1d.run(datavec, relativeError,
                                        startModel=kwargs.pop('startModel',
                                                              self.model),
                                        verbose=True, **kwargs)
            self.response1d = self.inv1d.response.array()

        if show:
            fig, ax = plt.subplots()
            drawModel1D(ax, np.diff(self.depth), self.model, color="blue",
                        plot='semilogx', label="inverted")
            ax = self.showSounding(amphi=False,
                                   response=self.response1d)

        return self.model

    def invertLine(self, line=None, nn=None, **kwargs):
        """Invert all soundings along a line."""
        if line is not None:
            nn = np.nonzero(self.line == line)[0]
        elif nn is None:
            nn = range(len(self.rx))

        self.MODELS = []
        # set up depth before
        if self.depth is None:
            self.depth = self.createDepthVector()

        self.allModels = np.ones([len(self.rx), len(self.depth)])
        model = 100
        for n in nn:
            self.setPos(n)
            model = self.invertSounding(startModel=30, show=False, **kwargs)
            self.MODELS.append(model)
            self.allModels[n, :] = model

        dx = np.sqrt(np.diff(self.rx[nn])**2 + np.diff(self.ry[nn])**2)
        self.xLine = np.cumsum(np.hstack((0., dx)))
        self.zLine = self.rz[nn]
        txm, tym = np.mean(self.tx), np.mean(self.ty)
        d2a = (self.rx[nn[0]] - txm)**2 + (self.ry[nn[0]] - tym)**2
        d2b = (self.rx[nn[-1]] - txm)**2 + (self.ry[nn[-1]] - tym)**2
        if d2b < d2a:  # line running towards transmitter
            self.xLine = self.xLine[-1] - self.xLine

        np.savez("models.npz", self.MODELS, self.xLine)
        self.showSection()

    def showSection(self, **kwargs):
        """Show all results along a line."""
        kwargs.setdefault("cMap", "Spectral")
        kwargs.setdefault("cMin", 1)
        kwargs.setdefault("cMax", 100)
        kwargs.setdefault("logScale", True)
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()

        # showStitchedModels(self.MODELS, ax=ax, x=self.xLine,  # pg>1.2.2
        #                    thk=np.diff(self.depth), **kwargs)
        THKMOD = [np.hstack((np.diff(self.depth), model))
                  for model in self.MODELS]  # obsolete with pg>1.2.2

        showStitchedModels(THKMOD, ax=ax, x=self.xLine,  # topo=self.zLine,
                           **kwargs)
        return ax

    def showDepthMap(self, **kwargs):
        """Show resistivity depth map."""
        pass

    def showSounding(self, nrx=None, position=None, response=None,
                     **kwargs):
        """Show amplitude and phase data."""
        cmp = kwargs.pop("cmp", self.cmp)
        if nrx is not None or position is not None:
            self.setPos(nrx, position)

        ax = kwargs.pop("ax", None)
        allcmp = ['x', 'y', 'z']
        if response is not None:
            respRe, respIm = np.reshape(response, (2, -1))
            respRe = np.reshape(respRe, (sum(cmp), -1))
            respIm = np.reshape(respIm, (sum(cmp), -1))

        ncmp = 0
        for i in range(3):
            if cmp[i] > 0:
                data = getattr(self, "data"+allcmp[i].upper())
                ax = showSounding(data, self.f, ax=ax, color="C"+str(i),
                                  marker="x", label="B"+allcmp[i], **kwargs)
                if response is not None:
                    ax[0].plot(respRe[ncmp], self.f, "-", color="C"+str(i))
                    ax[1].plot(respIm[ncmp], self.f, "-", color="C"+str(i))
                    ncmp += 1

        for a in ax:
            a.legend()

        return ax

    def showLineData(self, line=None, amphi=True, plim=[-180, 180],
                     ax=None, alim=None, log=False, **kwargs):
        """Show data of a line as pcolor."""
        cmp = kwargs.pop("cmp", self.cmp)
        if line is not None:
            nn = np.nonzero(self.line == line)[0]
        else:
            nn = np.arange(len(self.rx))

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2, squeeze=False,
                                   sharex=True, sharey=True)
        ncmp = 0
        allcmp = ['x', 'y', 'z']
        for i in range(3):
            if cmp[i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[:, nn]
                if amphi:
                    pc1 = ax[0, ncmp].matshow(np.log10(np.abs(data)),
                                              cmap="Spectral_r")
                    if alim is not None:
                        pc1.set_clim(alim)
                    pc2 = ax[1, ncmp].matshow(np.angle(data, deg=True),
                                              cMap="hsv")
                    pc2.set_clim(plim)
                else:
                    if log:
                        tol = 1e-3
                        if isinstance(symlog, float):
                            tol = symlog
                        pc1 = ax[0, ncmp].matshow(
                            symlog(np.real(data), tol), cmap="seismic")
                        if alim is not None:
                            aa = symlog(alim[0], tol)
                            pc1.set_clim([-aa, aa])
                        pc2 = ax[1, ncmp].matshow(
                            symlog(np.imag(data)), cmap="seismic")
                        if alim is not None:
                            aa = symlog(alim[1], tol)
                            pc2.set_clim([-aa, aa])
                    else:
                        pc1 = ax[0, ncmp].matshow(np.real(data), cmap="bwr")
                        if alim is not None:
                            pc1.set_clim([-alim[0], alim[0]])
                        pc2 = ax[1, ncmp].matshow(np.imag(data), cmap="bwr")
                        if alim is not None:
                            pc2.set_clim([-alim[1], alim[1]])

                divider = make_axes_locatable(ax[0, ncmp])
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(pc1, cax=cax, orientation="vertical")
                divider = make_axes_locatable(ax[1, ncmp])
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(pc2, cax=cax, orientation="vertical")
                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1

        ax[0, 0].set_ylim([-0.5, len(self.f)-0.5])
        # ax[0, 0].set_ylim(ax[0, 0].get_ylim()[::-1])
        yt = np.arange(0, len(self.f), 2)
        for aa in ax[:, 0]:
            aa.set_yticks(yt)
            aa.set_yticklabels(["{:.0f}".format(self.f[yy]) for yy in yt])
            aa.set_ylabel("f (Hz)")

        for a in ax.flat:
            a.set_aspect('auto')

        return ax

    def showField(self, field, **kwargs):
        """."""
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()

        background = kwargs.pop("background", None)
        ax.plot(self.rx, self.ry, "k.", ms=1, zorder=-10)
        ax.plot(self.tx, self.ty, "k*-", zorder=-1)
        if isinstance(field, str):
            kwargs.setdefault("label", field)
            field = getattr(self, field)

        ax, cb = plotSymbols(self.rx, self.ry, field, ax=ax, **kwargs)

        ax.set_aspect(1.0)
        x0 = np.floor(min(self.rx) / 1e4) * 1e4
        y0 = np.floor(min(self.ry) / 1e4) * 1e4
        ax.ticklabel_format(useOffset=x0, axis='x')
        ax.ticklabel_format(useOffset=y0, axis='y')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if background == "BKG":
            pg.viewer.mpl.underlayBKGMap(
                ax, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
        elif background is not None:
            pg.viewer.mpl.underlayMap(ax, self.utm, vendor=background)

        return ax, cb

    def showData(self, nf=0, ax=None, figsize=(9, 7), kwAmp={}, kwPhase={},
                 scale=0, **kwargs):
        """Show all three components as amp/phi or real/imag plots.

        Parameters
        ----------
        nf : int | float
            frequency index (int) or value (float) to plot
        """
        if isinstance(nf, float):
            nf = np.argmin(np.abs(self.f - nf))
            if self.verbose:
                print("Chose no f({:d})={:.0f} Hz".format(nf, self.f[nf]))

        overlay = kwargs.pop("overlay", True)
        amphi = kwargs.pop("amphi", True)
        if amphi:
            alim = kwargs.pop("alim", [-3, 0])
            plim = kwargs.pop("plim", [-180, 180])
            kwA = dict(cMap="Spectral_r", cMin=alim[0], cMax=alim[1],
                       radius=10, numpoints=0)
            kwA.update(kwAmp)
            kwP = dict(cMap="hsv", cMin=plim[0], cMax=plim[1],
                       radius=10, numpoints=0)
            kwP.update(kwPhase)
        else:
            log = kwargs.pop("log", True)
            alim = kwargs.pop("alim", [1e-3, 1])
            if log:
                alim[1] = symlog(alim[1], tol=alim[0])
            kwA = dict(cMap="seismic", radius=10, cMin=-alim[1], cMax=alim[1],
                       numpoints=0)
            kwA.update(kwAmp)
        if scale:
            amphi = False
            if self.prim is None:
                self.computePrimaryFields()

        allcmp = ["x", "y", "z"]
        # modify allcmp to show only subset
        if ax is None:
            fig, ax = plt.subplots(ncols=3, nrows=2,
                                   sharex=True, sharey=True, figsize=figsize)
        else:
            fig = ax.flat[0].figure

        for a in ax.flat:
            a.plot(self.tx, self.ty, "wx-", lw=2)
            a.plot(self.rx, self.ry, ".", ms=0, zorder=-10)
        for j, cmp in enumerate(allcmp):
            data = getattr(self, "DATA"+cmp.upper()).copy()
            if scale:
                data /= self.prim[j]
            if amphi:
                plotSymbols(self.rx, self.ry, np.log10(np.abs(data[nf])),
                            ax=ax[0, j], colorBar=(j == len(allcmp)-1), **kwA)
                plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                            ax=ax[1, j], colorBar=(j == len(allcmp)-1), **kwP)
                ax[0, j].set_title("log10 T"+cmp+" [nT/A]")
                ax[1, j].set_title(r"$\phi$"+cmp+" [°]")
            else:
                if log:
                    plotSymbols(self.rx, self.ry,
                                symlog(np.real(data[nf]), tol=alim[0]),
                                ax=ax[0, j], **kwA)
                    plotSymbols(self.rx, self.ry,
                                symlog(np.imag(data[nf]), tol=alim[0]),
                                ax=ax[1, j], **kwA)
                else:
                    plotSymbols(self.rx, self.ry, np.real(data[nf]),
                                ax=ax[0, j], **kwA)
                    plotSymbols(self.rx, self.ry, np.imag(data[nf]),
                                ax=ax[1, j], **kwA)

                ax[0, j].set_title("real T"+cmp+" [nT/A]")
                ax[1, j].set_title("imag T"+cmp+" [nT/A]")

        for a in ax.flat:
            a.set_aspect(1.0)
            a.plot(self.tx, self.ty, "k*-")
            # a.ticklabel_format(useOffset=550000, axis='x')
            # a.ticklabel_format(useOffset=5.78e6, axis='y')
            if overlay:
                try:
                    pg.viewer.mpl.underlayBKGMap(
                        a, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
                except Exception:
                    print("could not load BKG data")
                    overlay = False

        fig.suptitle("f="+str(self.f[nf])+"Hz")

        return fig, ax

    def computePrimaryFields(self):
        """Compute primary fields."""
        cfg = dict(self.cfg)
        fak = 4e-7 * np.pi * 1e9  # H->B and T in nT
        cfg["rec"] = [self.rx, self.ry, -self.alt, 0, 0]  # x
        self.pfx = bipole(res=[2e14], depth=[], xdirect=True,
                          freqtime=self.f, **cfg).real * fak
        cfg["rec"][3:5] = [90, 0]  # y
        self.pfy = bipole(res=[2e14], depth=[], xdirect=True,
                          freqtime=self.f, **cfg).real * fak
        cfg["rec"][3:5] = [0, 90]  # z
        self.pfz = bipole(res=[2e14], depth=[], xdirect=True,
                          freqtime=self.f, **cfg).real * fak
        self.prim = [self.pfx, self.pfy, self.pfz]

    def generateDataPDF(self, pdffile=None, linewise=False, **kwargs):
        """Generate a multi-page pdf file containing all data."""
        cmp = kwargs.pop("cmp", self.cmp)
        if linewise:
            pdffile = pdffile or self.basename + "-linedata.pdf"
        else:
            pdffile = pdffile or self.basename + "-data.pdf"

        plim = kwargs.pop("plim", [-90, 0])
        if kwargs.get("amphi", False):
            alim = kwargs.pop("alim", [1, 1])  # real/imag max
            kwargs.setdefault("log", True)
        else:
            alim = kwargs.pop("alim", [-2.5, 0])
        figsize = kwargs.pop("figsize", [9, 7])
        with PdfPages(pdffile) as pdf:
            if linewise:
                fig, ax = plt.subplots(figsize=figsize)
                self.showField(self.line, ax=ax, cMap="Spectral_r")
                ax.figure.savefig(pdf, format="pdf")
                fig, ax = plt.subplots(ncols=sum(cmp), nrows=2,
                                       figsize=figsize, squeeze=False,
                                       sharex=True, sharey=True)

                ul = np.unique(self.line)
                for li in ul[ul > 0]:
                    nn = np.nonzero(self.line == li)[0]
                    if np.isfinite(li) and len(nn) > 3:
                        self.showLineData(li, plim=plim,
                                          ax=ax, alim=alim, **kwargs)
                        fig.suptitle('line = {:.0f}'.format(li))
                        fig.savefig(pdf, format='pdf')  # bbox_inches="tight")
            else:
                fig, ax = plt.subplots(ncols=2, figsize=figsize, sharey=True)
                self.showField(np.arange(len(self.rx)), ax=ax[0],
                               cMap="Spectral_r")
                ax[0].set_title("Sounding number")
                self.showField(self.line, ax=ax[1], cMap="Spectral_r")
                ax[1].set_title("Line number")
                fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
                ax = None
                for i in range(len(self.f)):
                    fig, ax = self.showData(nf=i, ax=ax, figsize=figsize,
                                            **kwargs)
                    fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
                    plt.close(fig)
                    ax = None
                    # for a in ax.flat: a.cla()

    def generateModelPDF(self, pdffile=None, **kwargs):
        """Generate a PDF of all models."""
        dep = self.depth.copy()
        dep[:-1] += np.diff(self.depth) / 2
        pdffile = pdffile or self.basename + "-models5.pdf"
        kwargs.setdefault('cMin', 3)
        kwargs.setdefault('cMax', 200)
        kwargs.setdefault('logScale', True)
        with PdfPages(pdffile) as pdf:
            fig, ax = plt.subplots()
            for i in range(self.allModels.shape[1]):
                self.showField(self.allModels[:, i], ax=ax, **kwargs)
                ax.set_title('z = {:.1f}'.format(dep[i]))
                fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
                ax.cla()

    def saveData(self, fname=None, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.pop("cmp", self.cmp)
        if line is None:  # take all
            ind = np.nonzero(self.line > 0)[0]
        else:
            if line == "all":
                line = np.arange(1, max(self.line)+1)
            if hasattr(line, "__iter__"):
                for i in line:
                    self.saveData(line=i)
                return
            else:
                ind = np.nonzero(self.line == line)[0]

        allcmp = ['X', 'Y', 'Z']
        if fname is None:
            fname = self.basename
            if line is not None:
                fname += "-line" + str(line)

            for i in range(3):
                if cmp[i]:
                    fname += "B" + allcmp[i].lower()

        else:
            if fname.startswith("+"):
                fname = self.basename + "-" + fname

        meany = 0  # np.median(self.ry[ind])
        ypos = np.round(self.ry[ind]-meany)  # get them to a straight line
        rxpos = np.round(np.column_stack((self.rx[ind], ypos,
                                          self.rz[ind]-self.txAlt))*10)/10
        nF = len(self.f)
        nT = 1
        nR = rxpos.shape[0]
        nC = sum(cmp)
        DATA = []
        dataR = np.zeros([nT, nF, nR, nC])
        dataI = np.zeros([nT, nF, nR, nC])
        kC = 0
        Cmp = []
        for iC in range(3):
            if cmp[iC]:
                dd = -getattr(self, 'DATA'+allcmp[iC])[:, ind]
                dataR[0, :, :, kC] = dd.real
                dataI[0, :, :, kC] = dd.imag
                Cmp.append('B'+allcmp[iC].lower())
                kC += 1
        # error estimation
        absError = kwargs.pop("absError", 0.0015)
        relError = kwargs.pop("relError", 0.04)
        errorR = np.abs(dataR) * relError + absError
        errorI = np.abs(dataI) * relError + absError
        fak = 1  # 1e-9
        data = dict(dataR=dataR*fak, dataI=dataI*fak,
                    errorR=errorR*fak, errorI=errorI*fak,
                    tx_ids=[0], rx=rxpos, cmp=Cmp)
        DATA.append(data)
        # save them to NPY
        np.savez(fname+".npz",
                 tx=[np.column_stack((self.tx, self.ty-meany, self.tx*0))],
                 freqs=self.f,
                 DATA=DATA,
                 origin=self.origin,  # global coordinates with altitude
                 rotation=self.angle)


if __name__ == "__main__":
    txpos = np.array([[559497.46, 5784467.953],
                      [559026.532, 5784301.022]]).T
    self = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
    print(self)
    # self.generateDataPDF()
    self.showData(nf=1)
    # self.showField("alt", background="BKG")
    # self.invertSounding(nrx=20)
    # plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
    self.showSounding(nrx=20)
    # self.showData(nf=1)
    # self.generateDataPDF()
