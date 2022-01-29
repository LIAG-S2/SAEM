from glob import glob
import os.path
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

from .plotting import plotSymbols, showSounding, underlayBackground
from .modelling import fopSAEM, bipole


class CSEMData():
    """Class for CSEM frequency sounding."""

    def __init__(self, datafile=None, **kwargs):
        """Initialize CSEM data class

        Parameters
        ----------
        datafile : str
            data file to load if not None
        basename : str [datafile without extension]
            name for data (exporting, figures etc.)
        txPos : array
            transmitter position as polygone
        rx/ry/rz : iterable
            receiver positions
        f : iterable
            frequencies
        cmp : [int, int, int]
            active components
        alt : float
            flight altitude
        """
        self.basename = "noname"
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = kwargs.pop("cmp", [1, 0, 1])  # active components
        self.txAlt = kwargs.pop("txalt", 0.0)
        self.tx, self.ty = kwargs.pop("txPos", (None, None))
        self.rx = kwargs.pop("rx", np.array([100.0]))
        self.ry = kwargs.pop("ry", np.zeros_like(self.rx))
        self.f = kwargs.pop("f", [])
        self.rz = kwargs.pop("rz",
                             np.ones_like(self.rx) * kwargs.pop("alt", 0.0))
        self.line = np.ones_like(self.rx, dtype=int)
        self.alt = self.rz - self.txAlt
        self.depth = None
        self.prim = None
        self.DATA = None
        self.RESP = None
        self.ERR = None
        self.origin = [0, 0, 0]
        self.angle = 0
        self.radius = 10
        self.A = np.array([[1, 0], [0, 1]])
        if datafile is not None:
            self.loadData(datafile)

        self.basename = kwargs.pop("basename", self.basename)
        self.createConfig()

    def __repr__(self):
        """String representation of the class."""
        sdata = "CSEM data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))
        txlen = np.sqrt(np.diff(self.tx)**2+np.diff(self.ty)**2)[0]
        stx = "Transmitter length {:.0f}m".format(txlen)
        spos = "Sounding pos at " + (3*"{:1f},").format(*self.cfg["rec"][:3])

        return "\n".join((sdata, stx, spos))

    @property
    def nRx(self):
        """Number of receiver positions."""
        return len(self.rx)

    @property
    def nF(self):
        """Number of frequencies."""
        return len(self.f)

    def loadData(self, filename):
        """Load any data format."""
        if filename.endswith(".npz"):
            self.loadNpzFile(filename)
        elif filename.endswith(".mat"):
            self.loadMatFile(filename)

        self.DATA = np.stack([self.DATAX, self.DATAY, self.DATAZ])
        self.detectLines()
        self.radius = np.median(np.diff(self.rx))

    def loadNpzFile(self, filename):
        """Load data from numpy zipped file (inversion ready)."""
        ALL = np.load(filename, allow_pickle=True)
        freqs = ALL["freqs"]
        txgeo = ALL["tx"][0][:, :2].T
        data = ALL["DATA"][0]
        rxs = data["rx"]
        self.__init__(txPos=txgeo, f=freqs,
                      rx=rxs[:, 0], ry=rxs[:, 1], rz=rxs[:, 2])
        self.origin = ALL["origin"]
        self.angle = float(ALL["rotation"])
        self.basename = filename.replace(".npz", "")
        self.DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        self.DATAY = np.zeros_like(self.DATAX)
        self.DATAZ = np.zeros_like(self.DATAX)
        self.ERRX = np.ones_like(self.DATAX)
        self.ERRY = np.ones_like(self.DATAX)
        self.ERRZ = np.ones_like(self.DATAX)
        for ic, cmp in enumerate(data["cmp"]):
            setattr(self, "DATA"+cmp[1].upper(),
                    data["dataR"][0, ic, :, :] +
                    data["dataI"][0, ic, :, :] * 1j)
            setattr(self, "ERR"+cmp[1].upper(),
                    data["errorR"][0, ic, :, :] +
                    data["errorI"][0, ic, :, :] * 1j)

        self.cmp = [np.any(getattr(self, "DATA"+cc)) for cc in ["X", "Y", "Z"]]
        self.ERR = np.stack([self.ERRX, self.ERRY, self.ERRZ])

    def loadMatFile(self, filename):
        """Load data from mat file (WWU Muenster processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
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

        self.rx, self.ry = self.utm(MAT["lon"][0], MAT["lat"][0])
        self.f = np.squeeze(MAT["f"]) * 1.0
        self.DATAX = MAT["ampx"] * np.exp(MAT["phix"]*np.pi/180*1j)
        self.DATAY = MAT["ampy"] * np.exp(MAT["phiy"]*np.pi/180*1j)
        self.DATAY *= -1  # unless changed in the processing scripts
        self.DATAZ = MAT["ampz"] * np.exp(MAT["phiz"]*np.pi/180*1j)
        self.rz = MAT["alt"][0]
        self.alt = self.rz - self.txAlt

    def simulate(self, rho, thk, **kwargs):
        """Simulate data by assuming 1D layered model."""
        cmp = [1, 1, 1]  # cmp = kwargs.pop("cmp", self.cmp)
        self.createConfig()
        depth = np.hstack((0., np.cumsum(thk)))
        self.DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        self.DATAY = np.zeros_like(self.DATAX)
        self.DATAZ = np.zeros_like(self.DATAX)
        for ix in range(self.nRx):
            self.setPos(ix)
            fop1d = fopSAEM(depth, self.cfg, self.f, cmp)
            resp = fop1d.response(rho)

            respR, respI = np.reshape(resp, (2, -1))
            respC = np.reshape(respR+respI*1j, (3, -1))
            self.DATAX[:, ix] = respC[0, :]
            self.DATAY[:, ix] = respC[1, :]
            self.DATAZ[:, ix] = respC[2, :]

    def rotateBack(self):
        """Rotate coordinate system back to previously stored origin/angle."""
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

    def rotate(self, ang=None, line=None, origin=None):
        """Rotate positions and fields to a local coordinate system.

        Rotate the lines

        The origin and angle is stored so that original coordinates can be
        restored by rotateBack() which is called first (angle is global).

        Parameters
        ----------
        ang : float
            angle to rotate [otherwise determined from Tx or line positions]
        line : int [None]
            determine angle from specific line to be on x axis, if not given
            the angle is determined from Tx position so that it is on y axis
        origin : [float, float]
            origin of coordinate system, if not given, center of Tx
        """
        self.rotateBack()  # always go back to original system
        if origin is None:
            self.origin = [np.mean(self.tx), np.mean(self.ty)]
        else:
            self.origin = origin
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

    def rotatePositions(self, *args, **kwargs):  # backward compatibility
        self.rotate(*args, **kwargs)

    def detectLines(self, show=False):
        """Split data in lines for line-wise processing."""
        dt = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        dtmin = np.median(dt) * 2
        dx = np.round(np.diff(self.rx) / dt * 2)
        dy = np.round(np.diff(self.ry) / dt * 2)
        sdx = np.hstack((0, np.diff(np.sign(dx)), 0))
        sdy = np.hstack((0, np.diff(np.sign(dy)), 0))
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
            if i > 0 and dt[i-1] > dtmin:
                act = True
                nLine += 1

            if act:
                self.line[i] = nLine

        if show:
            self.showField(self.line)

    def removeNoneLineData(self):
        """Remove data not belonging to a specific line."""
        self.filter(nInd=np.nonzero(self.line)[0])

    def filter(self, f=-1, fmin=0, fmax=1e6, fInd=None, nInd=None):
        """Filter data according to frequency range and indices."""
        if fInd is None:
            bind = (self.f > fmin) & (self.f < fmax)  # &(self.f!=f)
            if f > 0:
                bind[np.argmin(np.abs(self.f - f))] = False

            fInd = np.nonzero(bind)[0]

        self.f = self.f[fInd]
        self.DATA = self.DATA[:, fInd, :]
        if np.any(self.RESP):
            self.RESP = self.RESP[:, fInd, :]

        if np.any(self.ERR):
            self.ERR = self.ERR[:, fInd, :]

        if np.any(self.prim):
            for i in range(3):
                self.prim[i] = self.prim[i][fInd, :]

        if nInd is not None:
            for tok in ['alt', 'rx', 'ry', 'rz', 'line']:
                setattr(self, tok, getattr(self, tok)[nInd])

            self.DATA = self.DATA[:, :, nInd]
            if hasattr(self, 'MODELS'):
                self.MODELS = self.MODELS[nInd, :]
            if self.prim is not None:
                for i in range(3):
                    self.prim[i] = self.prim[i][:, nInd]

        self.chooseData()  # make sure DATAX etc. have correct dimensions

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

    def showPos(self, ax=None, line=None, background=None):
        """Show positions."""
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.rx, self.ry, "b.", markersize=2)
        ax.plot(self.tx, self.ty, "r-", markersize=4)
        if hasattr(self, "nrx") and self.nrx < self.nRx:
            ax.plot(self.rx[self.nrx], self.ry[self.nrx], "bo", markersize=5)

        if line is not None:
            ax.plot(self.rx[self.line == line],
                    self.ry[self.line == line], "-", color="orange")

        ax.set_aspect(1.0)
        ax.grid(True)
        ax.set_xlabel("Easting (m) UTM32N")
        ax.set_ylabel("Northing (m) UTM32N")
        if background:
            underlayBackground(ax, background, self.utm)

        return ax

    def skinDepths(self, rho=30):
        """Compute skin depth based on a medium resistivity."""
        return np.sqrt(rho/self.f) * 500

    def createDepthVector(self, rho=30, nl=15):
        """Create depth vector."""
        sd = self.skinDepths(rho=rho)
        self.depth = np.hstack((0, pg.utils.grange(min(sd)*0.3, max(sd)*1.2,
                                                   n=nl, log=True)))
        # depth = np.hstack((0, np.cumsum(10**np.linspace(0.8, 1.5, 15))))
        # return depth

    def invertSounding(self, nrx=None, show=True, check=False, depth=None,
                       relError=0.03, absError=0.001, **kwargs):
        """Invert a single sounding."""
        kwargs.setdefault("verbose", False)
        if nrx is not None:
            self.setPos(nrx)

        if depth is not None:
            self.depth = depth
        if self.depth is None:
            self.createDepthVector()

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
            transData = pg.trans.TransSymLog(tol=1e-3)
            self.inv1d.transModel = transModel
            self.inv1d.transData = transData
            datavec = np.hstack((np.real(data), np.imag(data)))
            absoluteError = np.abs(datavec) * relError + absError
            relativeError = np.abs(absoluteError/datavec)
            self.model = self.inv1d.run(
                datavec, relativeError,
                startModel=kwargs.pop('startModel', self.model), **kwargs)
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
            self.createDepthVector()

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

    def chooseData(self, what="data", tol=1e-3):
        """Choose data to show by showData or showLineData.

        Parameters
        ----------
        what : str
            property or matrix to choose / show
                data - measured data
                error - data error (NOT READY)
                response - forward response (NOT READY)
                misfit - absolute misfit between data and response
                rmisfit - relative misfit between data and response
                wmisfit - error-weighted misfit
        """
        if what.lower() == "data":
            self.DATAX, self.DATAY, self.DATAZ = self.DATA
        elif what.lower() == "response":
            self.DATAX, self.DATAY, self.DATAZ = self.RESP
        elif what.lower() == "misfit":
            self.DATAX, self.DATAY, self.DATAZ = self.DATA - self.RESP
            # [self.DATAX, self.DATAY, self.DATAZ] = [
            #     self.DATA[i] - self.RESP[i] for i in range(3)]
        elif what.lower() == "rmisfit":
            # for i, DD in enumerate([self.DATAX, self.DATAY, self.DATAZ]):
            for i in range(3):
                rr = 1 - self.RESP[i].real / self.DATA[i].real
                ii = 1 - self.RESP[i].imag / self.DATA[i].imag
                rr[np.abs(rr) < tol] = 0
                ii[np.abs(ii) < tol] = 0
                # DD = rr + ii *1j
                if i == 0:
                    self.DATAX = rr + ii * 1j
                elif i == 1:
                    self.DATAY = rr + ii * 1j
                elif i == 2:
                    self.DATAZ = rr + ii * 1j
        elif what.lower() == "error":
            self.DATAX, self.DATAY, self.DATAZ = self.ERR
        elif what.lower() == "relerror":
            rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-6)
            ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-6)
            self.DATAX, self.DATAY, self.DATAZ = rr + ii * 1j
        elif what.lower() == "wmisfit":
            mis = self.DATA - self.RESP
            wmis = mis.real / self.ERR.real + mis.imag / self.ERR.imag * 1j
            self.DATAX, self.DATAY, self.DATAZ = wmis
        else:  # try using the argument?
            self.DATAX, self.DATAY, self.DATAZ = what

    def showLineData(self, line=None, amphi=True, plim=[-180, 180],
                     ax=None, alim=None, log=False, **kwargs):
        """Show data of a line as pcolor.

        Parameters
        ----------
        line : int
            line number to show. If not given all nonzero lines are used.
        cmp : [bool, bool, bool]
            components to plot. If not specified, use self.cmp
        amphi : bool [True]
            use (log10) amplitude and phase or real and imaginary part
        alim : [float, float]
            limits for the amplitude
        plim : [float, float]
            limits for the phase
        log : bool|float
            use logarithm (symlog) for amplitude and phase
            if float, this is the (white) tolerance
        """
        if "what" in kwargs:
            self.chooseData(kwargs["what"])

        cmp = kwargs.pop("cmp", self.cmp)
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2, squeeze=False,
                                   sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (10, 6)))
        ncmp = 0
        allcmp = ['x', 'y', 'z']
        for i in range(3):
            if cmp[i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[:, nn]
                if amphi:  # amplitud and phase
                    pc1 = ax[0, ncmp].matshow(np.log10(np.abs(data)),
                                              cmap="Spectral_r")
                    if alim is not None:
                        pc1.set_clim(alim)
                    pc2 = ax[1, ncmp].matshow(np.angle(data, deg=True),
                                              cMap="hsv")
                    pc2.set_clim(plim)
                else:  # real and imaginary part
                    if log:
                        tol = 1e-3
                        if isinstance(log, float):
                            tol = log
                        pc1 = ax[0, ncmp].matshow(
                            symlog(np.real(data), tol), cmap="seismic")
                        if alim is not None:
                            aa = symlog(alim[0], tol)
                            pc1.set_clim([-aa, aa])
                        pc2 = ax[1, ncmp].matshow(
                            symlog(np.imag(data), tol), cmap="seismic")
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

                for j, pc in enumerate([pc1, pc2]):
                    divider = make_axes_locatable(ax[j, ncmp])
                    cax = divider.append_axes("right", size="5%", pad=0.15)
                    cb = plt.colorbar(pc, cax=cax, orientation="vertical")
                    if not amphi:
                        if i == sum(cmp):
                            cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
                            cb.set_ticklabels(["-1e3", "-100", "-10", "+/-1p",
                                               "+10", "+100", "+1e3"])
                        else:
                            cb.set_ticks([])
                    else:
                        tit = "log10(B) in nT/A" if j == 0 else "phi in °"
                        cb.ax.set_title(tit)

                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1

        ax[0, 0].set_ylim([-0.5, len(self.f)-0.5])
        # ax[0, 0].set_ylim(ax[0, 0].get_ylim()[::-1])
        yt = np.arange(0, len(self.f), 2)
        ytl = ["{:.0f}".format(self.f[yy]) for yy in yt]
        for aa in ax[:, 0]:
            aa.set_yticks(yt)
            aa.set_yticklabels(ytl)
            aa.set_ylabel("f (Hz)")

        # xt = np.arange(0, len(nn), 10)
        xt = np.round(np.linspace(0, len(nn)-1, 7))
        xtl = ["{:.0f}".format(self.rx[nn[int(xx)]]) for xx in xt]
        for aa in ax[-1, :]:
            aa.set_xticks(xt)
            aa.set_xticklabels(xtl)
            aa.set_xlabel("x (m)")

        for a in ax.flat:
            a.set_aspect('auto')

        if "what" in kwargs:
            self.chooseData("data")

        return ax

    def showField(self, field, **kwargs):
        """Show any receiver-related field as color-coded rectangles/circles.

        Parameters
        ----------
        field : iterable | str
            field vector to plot or to extract from class, e.g. "line"
        cmap : mpl.colormap | str ["Spectral"]
            colormap
        colorBar : bool [True]
            draw colowbar
        cMin/cMax : float
            min/max values for colorbar
        logScale : bool [False]
            use logarithmic color scaling
        label : str
            label for the colorbar
        radius : float
            prescribing radius of symbol
        numpoints : int
            number of points (0 means circle)

        Returns
        -------
        ax, cb : matplotlib axes and colorbar instances
        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()

        kwargs.setdefault("radius", self.radius)
        background = kwargs.pop("background", None)
        ax.plot(self.rx, self.ry, "k.", ms=1, zorder=-10)
        ax.plot(self.tx, self.ty, "k*-", zorder=-1)
        if isinstance(field, str):
            kwargs.setdefault("label", field)
            field = getattr(self, field)

        ax, cb = plotSymbols(self.rx, self.ry, field, ax=ax, **kwargs)

        ax.set_aspect(1.0)
        x0 = np.floor(min(self.rx) / 1e4) * 1e4
        y0 = np.floor(np.median(self.ry) / 1e4) * 1e4
        if x0 > 100000 or y0 > 100000:
            ax.ticklabel_format(useOffset=x0, axis='x')
            ax.ticklabel_format(useOffset=y0, axis='y')

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if background:
            underlayBackground(ax, background, self.utm)

        return ax, cb

    def showData(self, nf=0, ax=None, figsize=(9, 7), kwAmp={}, kwPhase={},
                 scale=0, cmp=None, **kwargs):
        """Show all three components as amp/phi or real/imag plots.

        Parameters
        ----------
        nf : int | float
            frequency index (int) or value (float) to plot
        """
        if "what" in kwargs:
            self.chooseData(kwargs["what"])

        cmp = cmp or self.cmp
        if isinstance(nf, float):
            nf = np.argmin(np.abs(self.f - nf))
            if self.verbose:
                print("Chose no f({:d})={:.0f} Hz".format(nf, self.f[nf]))

        background = kwargs.pop("background", None)
        if background is not None and kwargs.pop("overlay", False):  # bwc
            background = "BKG"
        amphi = kwargs.pop("amphi", False)
        if amphi:
            alim = kwargs.pop("alim", [-3, 0])
            plim = kwargs.pop("plim", [-180, 180])
            kwA = dict(cMap="Spectral_r", cMin=alim[0], cMax=alim[1],
                       radius=self.radius, numpoints=0)
            kwA.update(kwAmp)
            kwP = dict(cMap="hsv", cMin=plim[0], cMax=plim[1],
                       radius=self.radius, numpoints=0)
            kwP.update(kwPhase)
        else:
            log = kwargs.pop("log", True)
            alim = kwargs.pop("alim", [1e-3, 1])
            if log:
                alim[1] = symlog(alim[1], tol=alim[0])
            kwA = dict(cMap="coolwarm", cMin=-alim[1], cMax=alim[1],
                       radius=self.radius, numpoints=0)
            kwA.update(kwAmp)
        if scale:
            amphi = False
            if self.prim is None:
                self.computePrimaryFields()

        allcmp = np.take(["x", "y", "z"], np.nonzero(cmp)[0])
        # modify allcmp to show only subset
        if ax is None:
            fig, ax = plt.subplots(ncols=len(allcmp), nrows=2, squeeze=False,
                                   sharex=True, sharey=True, figsize=figsize)
        else:
            fig = ax.flat[0].figure

        for a in ax.flat:
            a.plot(self.tx, self.ty, "wx-", lw=2)
            a.plot(self.rx, self.ry, ".", ms=0, zorder=-10)

        ncmp = 0
        for j, cc in enumerate(allcmp):
            data = getattr(self, "DATA"+cc.upper()).copy()
            if scale:
                data /= self.prim[j]
            if amphi:
                plotSymbols(self.rx, self.ry, np.log10(np.abs(data[nf])),
                            ax=ax[0, j], colorBar=(j == len(allcmp)-1), **kwA)
                plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                            ax=ax[1, j], colorBar=(j == len(allcmp)-1), **kwP)
                ax[0, j].set_title("log10 T"+cc+" [pT/A]")
                ax[1, j].set_title(r"$\phi$"+cc+" [°]")
            else:
                if log:
                    _, cb1 = plotSymbols(self.rx, self.ry,
                                      symlog(np.real(data[nf]), tol=alim[0]),
                                      ax=ax[0, j], **kwA)
                    _, cb2 = plotSymbols(self.rx, self.ry,
                                      symlog(np.imag(data[nf]), tol=alim[0]),
                                      ax=ax[1, j], **kwA)
                else:
                    _, cb1 = plotSymbols(self.rx, self.ry, np.real(data[nf]),
                                      ax=ax[0, j], **kwA)
                    _, cb2 = plotSymbols(self.rx, self.ry, np.imag(data[nf]),
                                      ax=ax[1, j], **kwA)

                ax[0, j].set_title("real T"+cc+" [nT/A]")
                ax[1, j].set_title("imag T"+cc+" [nT/A]")

                for cb in [cb1, cb2]:
                    if j == sum(cmp)-1:
                        cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
                        cb.set_ticklabels(["-1e3", "-100", "-10", "+/-1",
                                           "+10", "+100", "+1e3"])
                    else:
                        cb.set_ticks([])

            ncmp += 1

        for a in ax.flat:
            a.set_aspect(1.0)
            a.plot(self.tx, self.ty, "k*-")
            if background:
                underlayBackground(ax, background, self.utm)

        basename = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            basename += kwargs["what"]

        fig.suptitle(basename+"  f="+str(self.f[nf])+"Hz")

        if "what" in kwargs:
            self.chooseData("data")

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
        what = kwargs.pop("what", "data")
        self.chooseData(what)
        if linewise:
            pdffile = pdffile or self.basename + "-line-" + what + ".pdf"
        else:
            pdffile = pdffile or self.basename + "-" + what + ".pdf"

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
                                            alim=alim, **kwargs)
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
        ypos = np.round((self.ry[ind]-meany)*10)/10  # get to straight line
        rxpos = np.round(np.column_stack((self.rx[ind], ypos,
                                          self.rz[ind]-self.txAlt))*10)/10
        nF = len(self.f)
        nT = 1
        nR = rxpos.shape[0]
        nC = sum(cmp)
        DATA = []
        dataR = np.zeros([nT, nC, nF, nR])
        dataI = np.zeros_like(dataR)
        kC = 0
        Cmp = []
        for iC in range(3):
            if cmp[iC]:
                dd = getattr(self, 'DATA'+allcmp[iC])[:, ind]
                dataR[0, kC, :, :] = dd.real
                dataI[0, kC, :, :] = dd.imag
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
        np.savez(fname+".npz",
                 tx=[np.column_stack((self.tx, self.ty-meany, self.tx*0))],
                 freqs=self.f,
                 DATA=DATA,
                 origin=self.origin,  # global coordinates with altitude
                 rotation=self.angle)

    def loadResults(self, datafile=None, invmesh="Prisms", dirname=None,
                    jacobian=None):
        """Load inversion results from directory."""
        datafile = datafile or self.basename
        if dirname is None:
            dirname = datafile + "_" + invmesh + "/"
        if dirname[-1] != "/":
            dirname += "/"

        if os.path.exists(dirname + "inv_model.npy"):
            self.model = np.load(dirname + "inv_model.npy")
        else:
            self.model = np.load(sorted(glob(dirname+"sig_iter_*.npy"))[0])

        self.chi2s = np.loadtxt(dirname + "chi2.dat", usecols=3)
        respfiles = sorted(glob(dirname+"response_iter*.npy"))
        if len(respfiles) == 0:
            respfiles = sorted(glob(dirname+"reponse_iter*.npy"))  # TYPO
        if len(respfiles) == 0:
            pg.error("Could not find response file")

        response = np.load(respfiles[-1])
        respR, respI = np.split(response, 2)
        respC = respR + respI*1j
        ff = np.array([], dtype=bool)
        for i in range(3):
            if self.cmp[i]:
                tmp = self.DATA[i].ravel() * self.ERR[i].ravel()
                ff = np.hstack((ff, np.isfinite(tmp)))
        RESP = np.ones(np.prod([sum(self.cmp), self.nF, self.nRx]),
                       dtype=np.complex) * np.nan
        RESP[ff] = respC
        RESP = np.reshape(RESP, [sum(self.cmp), self.nF, self.nRx])
        self.RESP = np.ones((3, self.nF, self.nRx), dtype=np.complex) * np.nan
        self.RESP[np.nonzero(self.cmp)[0]] = RESP
        self.J = None
        if os.path.exists(dirname+"invmesh.vtk"):
            self.mesh = pg.load(dirname+"invmesh.vtk")
        else:
            self.mesh = pg.load(dirname + datafile + "_final_invmodel.vtk")
        print(self.mesh)
        jacobian = jacobian or datafile+"_jacobian.bmat"
        jname = dirname + jacobian
        if os.path.exists(jname):
            self.J = pg.load(jname)
            print("Loaded jacobian: "+jname, self.J.rows(), self.J.cols())
        elif os.path.exists(dirname+"jacobian.bmat"):
            self.J = pg.load(dirname+"jacobian.bmat")
            print("Loaded jacobian: ", self.J.rows(), self.J.cols())

    def showResult(self, **kwargs):
        """Show inversion result."""
        kwargs.setdefault("logScale", True)
        kwargs.setdefault("cMap", "Spectral")
        kwargs.setdefault("xlabel", "x (m)")
        kwargs.setdefault("ylabel", "z (m)")
        kwargs.setdefault("label", r"$\rho$ ($\Omega$m)")
        return pg.show(self.mesh, 1./self.model, **kwargs)

    def exportRxTxVTK(self, marker=1):
        """Export Receiver and Transmitter positions as VTK file."""
        rxmesh = pg.Mesh(3)
        for i in range(self.nRx):
            rxmesh.createNode([self.rx[i], self.ry[i], self.rz[i]], marker)

        rxmesh.exportVTK(self.basename+"-rxpos.vtk")
        txmesh = pg.Mesh(3)
        for xx, yy in zip(self.tx, self.ty):
            txmesh.createNode(xx, yy, self.txAlt)

        for i in range(txmesh.nodeCount()-1):
            txmesh.createEdge(txmesh.node(i), txmesh.node(i+1), marker)

        txmesh.exportVTK(self.basename+"-txpos.vtk")

    def showJacobianRow(self, iI=1, iC=0, iF=0, iR=0, cM=1.5, tol=1e-7,
                        save=False, **kwargs):
        """Show Jacobian row (model distribution for specific data).

        Parameters
        ----------
        iI : int [1]
            real (0) or imaginary (1) part
        iC : int
            component (out of existing ones!)
        iF : int [0]
            frequency index (into self.f)
        iR : int [0]
            receiver number
        cM : float [1.5]
            color scale maximum
        tol : float
            tolerance/threshold for symlog transformation

        **kwargs are passed to ps.show (e.g. cMin, )
        """
        allcmp = ['Bx', 'By', 'Bz']
        scmp = [allcmp[i] for i in np.nonzero(self.cmp)[0]]
        allP = ["Re ", "Im "]
        nD = self.J.rows() // 2
        nC = sum(self.cmp)
        nF = self.nF
        nR = self.nRx
        assert nD == nC * nF * nR, "Dimensions mismatch"
        iD = nD*iI + iC*(nF*nR) + iF*nR + iR
        Jrow = self.J.row(iD)
        sens = symlog(Jrow / self.mesh.cellSizes() * self.model, tol=tol)
        defaults = dict(cMap="bwr", cMin=-cM, cMax=cM, colorBar=False,
                        xlabel="x (m)", ylabel="z (m)")
        defaults.update(kwargs)
        ax, cb = pg.show(self.mesh, sens, **defaults)
        ax.plot(np.mean(self.tx), 5, "k*")
        ax.plot(self.rx[iR], 10, "kv")
        st = allP[iI] + scmp[iC] + ", f={:.0f}Hz, x={:.0f}m".format(
            self.f[iF], self.rx[iR])
        ax.set_title(st)
        fn = st.replace(" ", "_").replace(",", "").replace("=", "")
        if save:
            ax.figure.savefig("pics/"+fn+".pdf", bbox_inches="tight")

        return ax


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
