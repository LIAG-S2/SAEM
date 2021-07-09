from glob import glob
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import collections
from matplotlib.patches import Circle, RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns

import empymod
import pyproj

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D


def showSounding(snddata, freqs, ma="rx", ax=None, amphi=True, response=None):
    """Show amplitude and phase data."""
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True)

    if snddata.dtype == np.float:
        re = snddata[:len(snddata)//2]
        im = snddata[len(snddata)//2:]
        snddata = re + im * 1j
        print(len(freqs), len(data))

    if amphi:
        ax[0].loglog(np.abs(snddata), freqs, ma)
        ax[1].semilogy(np.angle(snddata)*180/np.pi, freqs, ma)
    else:
        ax[0].semilogy(np.real(snddata), freqs, ma)
        ax[1].semilogy(np.imag(snddata), freqs, ma)

    for a in ax:
        a.grid(True)

    if response is not None:
        showSounding(response, "b-", ax=ax, amphi=amphi)

    return ax


def plotSymbols(x, y, w, ax=None, cmap="Spectral",
                clim=None, radius=10, numpoints=0, colorBar=True):
    """Plot circles or rectangles for each point in a map.

    Parameters
    ----------
    x, y : iterable
        x and y positions
    w : iterable
        values to plot
    cmap : mpl.colormap | str
        colormap
    clim : (float, float)
        min/max values for colorbar
    radius : float
        prescribing radius of symbol
    numpoint : int
        number of points (0 means circle)
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", ms=0, zorder=-10)

    if clim is None:
        clim = [min(w), max(w)]

    patches = []
    for xi, yi in zip(x,y):
        if numpoints==0:
            rect = Circle( (xi,yi), radius, ec=None )
        else:
            rect = RegularPolygon((xi,yi), numpoints, radius=radius, ec=None)

        patches.append(rect)

    pc = collections.PatchCollection(patches, cmap=cmap, linewidths=0)
    pc.set_array(w)
    ax.add_collection(pc)
    pc.set_clim(clim)
    if colorBar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pc, cax=cax)

    return pc


def fwd(res, dep, inp, freqs):
    """Call empymods function bipole with the above arguments."""
    assert len(res) == len(dep), str(len(res)) + "/" + str(len(dep))
    OUT = empymod.bipole(res=np.concatenate(([2e14], res)),
                         depth=dep, freqtime=freqs, **inp)

    my = 4e-7 * np.pi
    OUT *=  my * 1e9

    return OUT

class myFwd(pg.Modelling):
    def __init__(self, depth, cfg, f, cmp=[0, 0, 1]):
        """Initialize the model."""
        super().__init__()
        self.dep = depth
        self.cfg = cfg
        self.cmp = cmp
        self.f = f
        self.mesh1d = pg.meshtools.createMesh1D(len(self.dep))
        self.setMesh(self.mesh1d)

    def response(self, model):
        """Forward response."""
        resp = []
        if self.cmp[0]:
            self.cfg['rec'][3:5] = (0, 0)
            resp.extend(fwd(model, self.dep, self.cfg, self.f))
        if self.cmp[1]:
            self.cfg['rec'][3:5] = (90, 0)
            resp.extend(fwd(model, self.dep, self.cfg, self.f))
        elif self.cmp[2]:
            self.cfg['rec'][3:5] = (0, 90)
            resp.extend(fwd(model, self.dep, self.cfg, self.f))

        return np.hstack((np.real(resp), np.imag(resp)))

    def createStartModel(self, data):
        return pg.Vector(len(self.dep), 100)





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

    def filter(self):
        pass

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
        fop = myFwd(depth_fixed, self.cfg, self.f, self.cmp)
        resistivity = np.ones_like(depth_fixed) * 100
        data = []
        for i, cmp in enumerate(["X", "Y", "Z"]):
            if self.cmp[i]:
                data.extend(getattr(self, "data"+cmp))

        inv = pg.Inversion()
        inv.setForwardOperator(fop)
        transModel = pg.trans.TransLog(1)
        inv.transModel = transModel
        datavec = np.hstack((np.real(data), np.imag(data)))
        absError = np.abs(datavec) * 0.03 + 0.001
        relError = np.abs(absError/datavec)
        model = inv.run(datavec, relError, startModel=100, verbose=True)
        if show:
            fig, ax = plt.subplots()
            drawModel1D(ax, np.diff(depth_fixed), model, color="blue",
                        plot='semilogx', label="inverted")
            ax = self.showSounding(data, amphi=False,
                                   response=inv.response.array())

    def showSounding(self, nrx=None, position=None, **kwargs):
        """Show amplitude and phase data."""
        if nrx is not None or position is not None:
            self.setPos(nrx, position)

        ax = None
        allcmp = ['x', 'y', 'z']
        for i in range(3):
            if self.cmp[i] > 0:
                data = getattr(self, "data"+allcmp[i].upper())
                ax = showSounding(data, self.f, ax=ax, ma="C"+str(i)+"x")

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
    self.generateDataPDF()
    # self.showData(nf=1)
    # self.showField("alt", background="BKG")
    # self.invertSounding(nrx=20)
    # plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
    sdfsdfsdf
    # self.cmp[0] = 1
    # self.cmp[1] = 1
    self.showSounding(nrx=20)
    # self.showData(nf=1)
    # self.generateDataPDF()


    sdfsdf
    # %%
    rtape = np.sqrt((rx-rx[0])**2 + (ry-ry[0])**2)
    # %%
    # nst = np.arange(7, len(rx))
    nst = np.arange(33)
    # nst = np.arange(10, 15)
    MODELS = []
    # model = resistivity
    for nrx in nst:
        setPos(nrx)
        data = DATA[:, nrx]
        fop = myFwd(depth_fixed)
        inv.setForwardOperator(fop)
        datavec = np.hstack((np.real(data), np.imag(data)))
        absError = np.abs(datavec) * 0.03 + 0.001
        relError = np.abs(absError/datavec)
        model = inv.run(datavec, relError, startModel=100,
                        robustData=False, verbose=True)
        MODELS.append(model.array())
    # %%
    xmid = rtape[nst]
    dx = np.diff(xmid)
    x = np.hstack((xmid[0]-dx[0]/2, xmid[:-1]+dx/2, xmid[-1]+dx[-1])) # - 130
    mod2D = np.array(MODELS[2:-2]).T.ravel()
    grid = pg.createGrid(x[2:-2], -np.hstack((depth_fixed, depth_fixed[-1]+10)))
    kw = dict(cMin=1, cMax=100, logScale=True, cmap="Spectral", colorBar=True)
    ax, cb = pg.show(grid, mod2D, **kw)
    # ax.set_xlim(x[1], x[-1])
    # ax.set_ylim(-200, 0)
    ax.figure.savefig("flight2line1-result2.pdf", bbox_inches="tight")
    # %%
    grid.save("saem.bms")
    np.savetxt("saem.vec", mod2D)
    # %
    # %%
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].matshow(np.log10(np.abs(DATA)))
    ax[0, 1].matshow(np.angle(DATA))

