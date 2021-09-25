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

from .plotting import plotSymbols, showSounding
from .modelling import fopSAEM, bipole


class CSEMData():
    """Class for CSEM frequency sounding."""

    def __init__(self, **kwargs):
        self.basename = "noname"
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = [0, 0, 1]  # active components
        self.txAlt = kwargs.pop("txalt")
        self.tx, self.ty = kwargs.pop("txPos", (None, None))
        self.depth = None
        self.prim = None
        if "datafile" in kwargs:
            self.loadData(kwargs["datafile"])

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
        filenames = glob(filename)
        assert len(filenames) > 0
        filename = filenames[0]
        self.basename = filename.replace(".mat", "")
        MAT = loadmat(filename)
        for filename in filenames[1:]:
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

    def detectLines(self, show=False):
        """Split data in lines for line-wise processing."""
        dx = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        sdx = np.hstack((0, np.diff(np.sign(np.diff(self.rx))), 0))
        sdy = np.hstack((0, np.diff(np.sign(np.diff(self.ry))), 0))
        self.line = np.ones_like(self.rx) * np.nan
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

    def filter(self, fmin=0, fmax=1e6, f=-1, nInd=None, nMin=None, nMax=None):
        """Filter data according ."""
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

    def setPos(self, nrx, position=None, show=False):
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

    def showPos(self, ax=None):
        """Show positions."""
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.rx, self.ry, "b.", markersize=2)
        ax.plot(self.tx, self.ty, "r-", markersize=4)
        if hasattr(self, "nrx"):
            ax.plot(self.rx[self.nrx], self.ry[self.nrx], "bo", markersize=5)

        ax.set_aspect(1.0)
        ax.grid(True)

    def invertSounding(self, nrx=None, show=True, check=False, depth=None,
                       relError=0.03, absError=0.001, **kwargs):
        """Invert a single sounding."""
        if nrx is not None:
            self.setPos(nrx)

        if self.depth is None:
            if depth is not None:
                self.depth = depth
            else:
                self.depth = np.hstack(
                    (0, np.cumsum(10**np.linspace(0.8, 1.5, 15))))

        self.fop1d = fopSAEM(self.depth, self.cfg, self.f, self.cmp)
        self.fop1d.modelTrans.setLowerBound(1.0)
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

    def invertLine(self, nn=None, line=None, **kwargs):
        """Invert all soundings along a line."""
        if line is not None:
            nn = np.nonzero(self.line == line)[0]
        elif nn is None:
            nn = range(len(self.rx))

        self.MODELS = []
        # set up depth before
        self.depth = np.hstack((0, np.cumsum(10**np.linspace(0.8, 1.5, 15))))
        self.allModels = np.ones([len(self.rx), len(self.depth)])
        model = 100
        for n in nn:
            self.setPos(n)
            model = self.invertSounding(startModel=30, show=False)
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

    def showSounding(self, nrx=None, position=None, response=None,
                     **kwargs):
        """Show amplitude and phase data."""
        if nrx is not None or position is not None:
            self.setPos(nrx, position)

        ax = kwargs.pop("ax", None)
        allcmp = ['x', 'y', 'z']
        if response is not None:
            respRe, respIm = np.reshape(response, (2, -1))
            respRe = np.reshape(respRe, (sum(self.cmp), -1))
            respIm = np.reshape(respIm, (sum(self.cmp), -1))

        ncmp = 0
        for i in range(3):
            if self.cmp[i] > 0:
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

    def showLineData(self, line=None, amphi=True):
        """Show data of a line as pcolor."""
        if line is not None:
            nn = np.nonzero(self.line == line)[0]
        else:
            nn = np.arange(len(self.rx))

        fig, ax = plt.subplots(ncols=sum(self.cmp), nrows=2,
                               sharex=True, sharey=True, squeeze=False)
        ncmp = 0
        allcmp = ['x', 'y', 'z']
        for i in range(3):
            if self.cmp[i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[:, nn]
                if amphi:
                    pc1 = ax[0, ncmp].matshow(np.log10(np.abs(data)),
                                              cmap="Spectral_r")
                    pc2 = ax[1, ncmp].matshow(np.angle(data, deg=True),
                                              cMap="hsv")
                    pc2.set_clim([-180, 180])
                else:
                    pc1 = ax[0, ncmp].matshow(np.real(data),
                                              cmap="Spectral_r")
                    pc2 = ax[1, ncmp].matshow(np.real(data),
                                              cmap="Spectral_r")

                divider = make_axes_locatable(ax[0, ncmp])
                cax = divider.append_axes("bottom", size="15%", pad=0.15)
                plt.colorbar(pc1, cax=cax, orientation="horizontal")
                divider = make_axes_locatable(ax[1, ncmp])
                cax = divider.append_axes("bottom", size="15%", pad=0.15)
                plt.colorbar(pc2, cax=cax, orientation="horizontal")
                ncmp += 1

        ax[0, 0].set_ylim(ax[0, 0].get_ylim()[::-1])

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
            field = getattr(self, field)

        plotSymbols(self.rx, self.ry, field, ax=ax, **kwargs)

        ax.set_aspect(1.0)
        x0 = np.floor(min(self.rx) / 1e4) * 1e4
        y0 = np.floor(min(self.ry) / 1e4) * 1e4
        ax.ticklabel_format(useOffset=x0, axis='x')
        ax.ticklabel_format(useOffset=y0, axis='y')
        if background == "BKG":
            pg.viewer.mpl.underlayBKGMap(
                ax, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
        elif background is not None:
            pg.viewer.mpl.underlayMap(ax, self.utm, vendor=background)

        return ax

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

        kwA = dict(cMap="Spectral_r", radius=10, cMin=-3, cMax=1, numpoints=0)
        kwA.update(kwAmp)
        kwP = dict(cMap="hsv", radius=10, cMin=-180, cMax=180, numpoints=0)
        kwP.update(kwPhase)
        amphi = kwargs.pop("amphi", True)
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
                plotSymbols(self.rx, self.ry, np.real(data[nf]),
                            ax=ax[0, j])
                plotSymbols(self.rx, self.ry, np.imag(data[nf]),
                            ax=ax[1, j])
                ax[0, j].set_title("real T"+cmp+" [nT/A]")
                ax[1, j].set_title("imag T"+cmp+" [nT/A]")

        for a in ax.flat:
            a.set_aspect(1.0)
            a.ticklabel_format(useOffset=550000, axis='x')
            a.ticklabel_format(useOffset=5.78e6, axis='y')
            pg.viewer.mpl.underlayBKGMap(
                a, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')

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

    def generateDataPDF(self, pdffile=None, **kwargs):
        """Generate a multi-page pdf file containing all data."""
        pdffile = pdffile or self.basename + "-data.pdf"
        with PdfPages(pdffile) as pdf:
            fig, ax = plt.subplots(ncols=2, figsize=(9, 7), sharey=True)
            self.showField(np.arange(len(self.rx)), ax=ax[0],
                           cMap="Spectral_r")
            ax[0].set_title("Sounding number")
            self.showField(self.line, ax=ax[1], cMap="Spectral_r")
            ax[1].set_title("Line number")
            fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
            ax = None
            for i in range(len(self.f)):
                fig, ax = self.showData(nf=i, ax=ax)
                fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
                for a in ax.flat:
                    a.cla()


if __name__ == "__main__":
    # import transmitter (better by kmlread)
    txpos = np.array([[559497.46, 5784467.953],
                      [559026.532, 5784301.022]]).T
    self = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
    print(self)
    # self.generateDataPDF()
    self.showData(nf=1)
    # self.showField("alt", background="BKG")
    # self.invertSounding(nrx=20)
    # plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
    # self.cmp[0] = 1
    # self.cmp[1] = 1
    self.showSounding(nrx=20)
    # self.showData(nf=1)
    # self.generateDataPDF()
