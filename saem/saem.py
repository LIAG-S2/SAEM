from glob import glob
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pyproj

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D

from .plotting import plotSymbols, showSounding
from .modelling import fopSAEM



class CSEMData():
    """Class for CSEM frequency sounding."""
    def __init__(self, **kwargs):
        self.basename = "noname"
        zone = kwargs.pop("zone", 32)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = [0, 0, 1]  # active components
        self.txAlt = kwargs.pop("txalt")
        self.tx, self.ty = kwargs.pop("txPos", (None, None))
        if "datafile" in kwargs:
            self.loadData(kwargs["datafile"])

    def __repr__(self):
        """String representation of the class."""
        sdata = "CSEM data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))
        txlen = np.sqrt(np.diff(self.tx)**2+np.diff(self.ty)**2)[0]
        stx= "Transmitter length {:.0f}m".format(txlen)
        spos = "Sounding pos at " +(3*"{:1f},").format(*self.cfg["rec"][:3])

        return "\n".join((sdata, stx, spos))

    def loadData(self, filename):
        """Load data from mat file."""
        filenames = glob(filename)
        assert len(filenames) > 0
        filename = filenames[0]
        self.basename = filename.replace(".mat", "")
        print(filename)
        MAT = loadmat(filename)
        for filename in filenames[1:]:
            MAT1 = loadmat(filename)
            assert len(MAT["f"]) == len(MAT1["f"]), filename+" nf not matching"
            assert np.allclose(MAT["f"], MAT1["f"]), filename+" f not matching"
            for key in MAT1.keys():
                if key[0] != "_" and key != "f":
                    # print(key, MAT[key].shape, MAT1[key].shape)
                    MAT[key] = np.hstack([MAT[key], MAT1[key]])

        self.f = np.squeeze(MAT["f"])
        self.rx, self.ry = self.utm(MAT["lon"][0], MAT["lat"][0])
        self.DATAX = MAT["ampx"] * np.exp(MAT["phix"]*np.pi/180*1j)
        self.DATAY = MAT["ampy"] * np.exp(MAT["phiy"]*np.pi/180*1j)
        self.DATAZ = MAT["ampz"] * np.exp(MAT["phiz"]*np.pi/180*1j)
        self.alt = MAT["alt"][0] - self.txAlt
        self.createConfig()

    def filter(self, fmin=0, fmax=1e6, f=-1):
        """Filter data according ."""
        ind = np.nonzero((self.f>fmin)&(self.f<fmax)&(self.f!=f))[0]
        self.f = self.f[ind]
        self.DATAX = self.DATAX[ind, :]
        self.DATAY = self.DATAY[ind, :]
        self.DATAZ = self.DATAZ[ind, :]

    def mask(self):
        pass

    def createConfig(self):
        """Create EMPYMOD input argument configuration."""
        self.cfg = {'src': [self.tx[0], self.tx[1], self.ty[0], self.ty[1],
                             0.1, 0.1], 'strength': 1, 'mrec': True,
                     'rec': [self.rx[0], self.ry[0], -self.alt[0], 0, 90],
                     'srcpts': 11, 'htarg': {'pts_per_dec': -1}, 'verb': 1}
        # self.inpX['rec'][3:5] = (0, 0)  # x direction, dip 0, azimuth zero
        # self.inpY['rec'][3:5] = (90, 0)  # y direction, dip 0, azimuth 90°

    def setPos(self, nrx, position=None):
        """The ."""
        if position:
            dr = (self.rx - position[0])**2 + (self.ry - position[1])**2
            print("distance is ", np.sqrt(dr))
            nrx = np.argmin(dr)

        self.cfg["rec"][:3] = self.rx[nrx], self.ry[nrx], -self.alt[nrx]
        self.dataX = self.DATAX[:, nrx]
        self.dataY = self.DATAY[:, nrx]
        self.dataZ = self.DATAZ[:, nrx]

    def invertSounding(self, nrx=None, show=True):
        """Invert a single sounding."""
        if nrx is not None:
            self.setPos(nrx)

        depth_fixed = np.concatenate((np.arange(0, 30, 2.5),
                                      np.arange(30, 100., 10),
                                      np.arange(100, 300., 25)))
        fop = fopSAEM(depth_fixed, self.cfg, self.f, self.cmp)
        if 0:
            model = np.ones_like(depth_fixed) * 100
            self.response1d = fop(model)
        else:
            data = []
            for i, cmp in enumerate(["X", "Y", "Z"]):
                if self.cmp[i]:
                    data.extend(getattr(self, "data"+cmp))

            self.inv1d = pg.Inversion()
            self.inv1d.setForwardOperator(fop)
            transModel = pg.trans.TransLog(1)
            self.inv1d.transModel = transModel
            datavec = np.hstack((np.real(data), np.imag(data)))
            absError = np.abs(datavec) * 0.03 + 0.001
            relError = np.abs(absError/datavec)
            model = self.inv1d.run(datavec, relError, startModel=100, verbose=True)
            self.response1d = self.inv1d.response.array()
        if show:
            fig, ax = plt.subplots()
            drawModel1D(ax, np.diff(depth_fixed), model, color="blue",
                        plot='semilogx', label="inverted")
            ax = self.showSounding(amphi=False,
                                   response=self.response1d)

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

    def showField(self, field, **kwargs):
        """."""
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            fig, ax = plt.subplots()

        background = kwargs.pop("background", None)
        ax.plot(self.rx, self.ry, ".", ms=0, zorder=-10)
        if isinstance(field, str):
            field = getattr(self, field)

        plotSymbols(self.rx, self.ry, field, ax=ax, **kwargs)

        ax.set_aspect(1.0)
        ax.ticklabel_format(useOffset=550000, axis='x')
        ax.ticklabel_format(useOffset=5.78e6, axis='y')
        if background == "BKG":
            pg.viewer.mpl.underlayBKGMap(
                ax, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
        elif background is not None:
            pg.viewer.mpl.underlayMap(ax, self.utm, vendor=background)

    def showData(self, nf=0, ax=None, figsize=(9, 7), kwAmp={}, kwPhase={}):
        """Generate a multi-page pdf file containing all data."""
        kwA = dict(cmap="Spectral_r", radius=10, clim=(-3, 1), numpoints=0)
        kwA.update(kwAmp)
        kwP = dict(cmap="hsv", radius=10, clim=(-180, 180), numpoints=0)
        kwP.update(kwPhase)
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
            data = getattr(self, "DATA"+cmp.upper())
            plotSymbols(self.rx, self.ry, np.log10(np.abs(data[nf])),
                        ax=ax[0, j], colorBar=(j==len(allcmp)-1), **kwA)
            plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                        ax=ax[1, j], colorBar=(j==len(allcmp)-1), **kwP)
            ax[0, j].set_title("log10 T"+cmp+" [nT/A]")

        for a in ax.flat:
            a.set_aspect(1.0)
            a.ticklabel_format(useOffset=550000, axis='x')
            a.ticklabel_format(useOffset=5.78e6, axis='y')
            pg.viewer.mpl.underlayBKGMap(
                a, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')

        fig.suptitle("f="+str(self.f[nf])+"Hz")

        return fig, ax


    def generateDataPDF(self, pdffile=None, **kwargs):
        """Generate a multi-page pdf file containing all data."""
        pdffile = pdffile or self.basename + "-data.pdf"
        ax = None
        with PdfPages(pdffile) as pdf:
            for i in range(len(self.f)):
                fig, ax = self.showData(nf=i, ax=ax)
                fig.savefig(pdf, format='pdf', bbox_inches="tight")
                for a in ax.flat:
                    a.cla()

if __name__ == "__main__":
    # %% import transmitter (better by kmlread)
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


