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
from matplotlib.colors import LogNorm, SymLogNorm

from .plotting import plotSymbols, showSounding, updatePlotKwargs
from .plotting import underlayBackground, makeSymlogTicks, dMap
from .modelling import fopSAEM, bipole


class CSEMData():
    """Class for CSEM frequency-domain data patch (single Tx)."""

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
        self.isLoop = False
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = kwargs.pop("cmp", [1, 1, 1])  # active components
        self.txAlt = kwargs.pop("txalt", 0.0)
        self.tx, self.ty = kwargs.pop("txPos", (None, None))
        self.rx = kwargs.pop("rx", np.array([100.0]))
        self.ry = kwargs.pop("ry", np.zeros_like(self.rx))
        self.f = kwargs.pop("f", [])
        self.rz = kwargs.pop("rz",
                             np.ones_like(self.rx) * kwargs.pop("alt", 0.0))
        self.line = kwargs.pop("line", np.ones_like(self.rx, dtype=int))
        self.alt = self.rz - self.txAlt
        self.depth = None
        self.prim = None
        self.DATA = np.zeros((3, self.nF, self.nRx), dtype=complex)
        self.RESP = None
        self.ERR = None
        self.origin = [0, 0, 0]
        self.angle = 0
        self.llthres = 1e-3
        self.A = np.array([[1, 0], [0, 1]])
        if datafile is not None:
            self.loadData(datafile)

        dxy = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        self.radius = np.median(dxy) * 0.5
        self.basename = kwargs.pop("basename", self.basename)
        self.chooseData("data")
        if self.tx is not None:
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

    def loadData(self, filename, detectLines=False):
        """Load any data format."""
        if filename.endswith(".npz"):
            self.loadNpzFile(filename)
        elif filename.endswith(".mat"):
            if not self.loadMatFile(filename):
                print("No frequency in data, try read BGR style")
                self.loadMatFile2(filename)

        if len(self.line) != len(self.rx):
            self.line = np.ones_like(self.rx, dtype=int)

        if detectLines:
            self.detectLines()

    def addData(self, new, detectLines=False):
        """Add (concatenate) data."""
        if isinstance(new, str):
            new = CSEMData(new)
        if new.tx is not None:
            assert np.allclose(self.tx, new.tx), "Tx(x) not matching!"
        if new.ty is not None:
            assert np.allclose(self.ty, new.ty), "Tx(y) not matching!"
        if new.f is not None:
            assert np.allclose(self.f, new.f)
        for attr in ["rx", "ry", "rz", "line", "alt",
                     'DATA', 'ERR', 'RESP', 'prim']:
            one = getattr(self, attr)
            two = getattr(new, attr)
            if np.any(one) and np.any(two):
                setattr(self, attr, np.concatenate((one, two), axis=-1))

    def loadNpzFile(self, filename, nr=0):
        """Load data from numpy zipped file (inversion ready)."""
        ALL = np.load(filename, allow_pickle=True)
        self.basename = filename.replace(".npz", "")
        self.extractData(ALL, nr=nr)
        self.origin = ALL["origin"]
        self.angle = float(ALL["rotation"])
        return True

    def extractData(self, ALL, nr=0):
        """Extract data from NPZ structure."""
        freqs = ALL["freqs"]
        txgeo = ALL["tx"][nr][:, :2].T
        data = ALL["DATA"][nr]
        rxs = data["rx"]
        self.__init__(txPos=txgeo, f=freqs,
                      rx=rxs[:, 0], ry=rxs[:, 1], rz=rxs[:, 2])

        if 'line' in ALL:
            self.line = ALL["line"]

        self.DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        self.DATAY = np.zeros_like(self.DATAX)
        self.DATAZ = np.zeros_like(self.DATAX)
        try:
            self.cmp = ALL["cmp"]
        except Exception:
            print('CMP detect change exception, using old way')
            self.cmp = [np.any(getattr(self, "DATA"+cc)) for cc in ["X", "Y", "Z"]]

        self.ERRX = np.ones_like(self.DATAX)
        self.ERRY = np.ones_like(self.DATAY)
        self.ERRZ = np.ones_like(self.DATAZ)
        for ic, cmp in enumerate(data["cmp"]):
            setattr(self, "DATA"+cmp[1].upper(),
                    data["dataR"][0, ic, :, :] +
                    data["dataI"][0, ic, :, :] * 1j)
            setattr(self, "ERR"+cmp[1].upper(),
                    data["errorR"][0, ic, :, :] +
                    data["errorI"][0, ic, :, :] * 1j)
        self.DATA = np.stack([self.DATAX, self.DATAY, self.DATAZ])
        self.ERR = np.stack([self.ERRX, self.ERRY, self.ERRZ])

    def loadMatFile(self, filename):
        """Load data from mat file (WWU Muenster processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
        print(filenames)
        assert len(filenames) > 0
        filename = filenames[0]
        MAT = loadmat(filename)
        if "f" not in MAT:
            return False
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
        self.DATA = np.stack([self.DATAX, self.DATAY, self.DATAZ])
        return True

    def loadMatFile2(self, filename):
        """Load data from mat file (Olaf BGR processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
        assert len(filenames) > 0
        filename = filenames[0]
        MAT = loadmat(filename)["ztfs"][0][0]
        if len(filenames) > 1:
            print("read "+filename)
        for filename in filenames[1:]:
            print("reading "+filename)
            MAT1 = loadmat(filename)["ztfs"][0][0]
            for i in range(len(MAT)):
                if i == 12:
                    assert len(MAT[i]) == len(MAT1[i]), "nF not matching"
                    assert np.allclose(MAT[i], MAT1[i]), "freqs not matching"
                else:
                    MAT[i] = np.concatenate((MAT[i], MAT1[i]), axis=-1)

        self.f = np.round(100.0 / np.squeeze(MAT[12])) / 100.
        self.ry, self.rx = MAT[19]
        self.rz = MAT[18][0]
        DY, DX, DZ = np.squeeze(MAT[13])
        self.DATA = np.stack((-DX, -DY, DZ))
        self.ERR = np.squeeze(MAT[14])
        self.alt = self.rz - self.txAlt
        return True

    def simulate(self, rho, thk=[], **kwargs):
        """Simulate data by assuming 1D layered model."""
        cmp = [1, 1, 1]  # cmp = kwargs.pop("cmp", self.cmp)
        self.createConfig()
        rho = np.atleast_1d(rho)
        thk = np.atleast_1d(thk)
        if len(thk) > 0:
            assert len(rho) == len(thk) + 1, "rho/thk lengths do not match"
            depth = np.hstack((0., np.cumsum(thk)))
        else:  # append an artificial layer to enforce RHS
            rho = np.hstack((rho, rho[0]))
            depth = np.hstack((0., 1000.))

        DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        DATAY = np.zeros_like(DATAX)
        DATAZ = np.zeros_like(DATAX)
        for ix in range(self.nRx):
            self.setPos(ix)
            fop1d = fopSAEM(depth, self.cfg, self.f, cmp)
            resp = fop1d.response(rho)

            respR, respI = np.reshape(resp, (2, -1))
            respC = np.reshape(respR+respI*1j, (3, -1))
            DATAX[:, ix] = respC[0, :]
            DATAY[:, ix] = respC[1, :]
            DATAZ[:, ix] = respC[2, :]

        self.RESP = np.stack([DATAX, DATAY, DATAZ])
        if kwargs.pop("show", False):
            if len(np.unique(self.line)) == 1:
                self.showLineData(what="response", **kwargs)
            else:
                self.showData(what="response")

    def rotateBack(self):
        """Rotate coordinate system back to previously stored origin/angle."""
        self.tx, self.ty = self.A.T.dot(np.array([self.tx, self.ty]))
        self.rx, self.ry = self.A.T.dot(np.vstack([self.rx, self.ry]))
        self.tx += self.origin[0]
        self.ty += self.origin[1]
        self.rx += self.origin[0]
        self.ry += self.origin[1]
        self.origin = [0, 0, 0]
        self.angle = 0
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

    def setOrigin(self, origin):
        """Set origin."""
        self.tx -= origin[0]
        self.ty -= origin[1]
        self.rx -= origin[0]
        self.ry -= origin[1]
        self.origin = origin

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

    def detectLinesByDistance(self, axis='x', sort=True, show=False,
                              minDist=200.):
        """Alernative - Split data in lines for line-wise processing."""

        dummy = np.zeros_like(self.rx, dtype=int)
        self.line = np.zeros_like(self.rx, dtype=int)
        li = 0
        for ri in range(1, len(self.rx)):
            dummy[ri-1] = li
            dist = np.sqrt((self.rx[ri]-self.rx[ri-1])**2 +\
                           (self.ry[ri]-self.ry[ri-1])**2)
            if dist > minDist:
                li += 1
        dummy[-1] = li

        if sort:
            means = []
            for li in np.unique(dummy):
                if axis == 'x':
                    means.append(np.mean(self.ry[dummy==li], axis=0))
                elif axis == 'y':
                    means.append(np.mean(self.rx[dummy==li], axis=0))
            lsorted = np.argsort(means)
            for li, lold in enumerate(lsorted):
                self.line[dummy==lold] = li + 1

        if show:
            self.showField(self.line)

    def detectLinesAlongAxis(self, axis='x', sort=True, show=False):
        """Alernative - Split data in lines for line-wise processing."""

        if axis == 'x':
            r = self.rx
        elif axis == 'y':
            r = self.ry
        else:
            print('Choose either *x* or *y* axis. Aborting this method ...')
            return

        dummy = np.zeros_like(self.rx, dtype=int)
        self.line = np.zeros_like(self.rx, dtype=int)
        li = 0
        last_sign = np.sign(r[1] - r[0])
        for ri in range(1, len(self.rx)):
            sign = np.sign(r[ri] - r[ri-1])
            dummy[ri-1] = li
            if sign != last_sign:
                li += 1
                last_sign *= -1
        dummy[-1] = li

        if sort:
            means = []
            for li in np.unique(dummy):
                if axis == 'x':
                    means.append(np.mean(self.ry[dummy==li], axis=0))
                elif axis == 'y':
                    means.append(np.mean(self.rx[dummy==li], axis=0))
            lsorted = np.argsort(means)
            for li, lold in enumerate(lsorted):
                self.line[dummy==lold] = li + 1

        if show:
            self.showField(self.line)

    def removeNoneLineData(self):
        """Remove data not belonging to a specific line."""
        self.filter(nInd=np.nonzero(self.line)[0])

    def txDistance(self):
        """Distance to transmitter."""
        ang = np.median(np.arctan2(np.diff(self.ty), np.diff(self.tx)))
        ang += np.pi / 2
        A = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        rx, ry = A.dot(np.array([self.rx-np.mean(self.tx),
                                 self.ry-np.mean(self.ty)]))
        return np.abs(rx)

    def filter(self, f=-1, fmin=0, fmax=1e6, fInd=None, nInd=None,
               minTxDist=None, maxTxDist=None, every=None):
        """Filter data according to frequency and and receiver properties.

        Parameters
        ----------
        f : float, optional
            frequency (next available) to remove from data
        fmin : float [0]
            minimum frequency to keep
        fmax : float [9e99]
            minimum frequency to keep
        fInd : iterable, optional
            index array of frequencies to use
        nInd : iterable, optional
            index array for receivers to use, alternatively
        minTxDist : float
            minimum distance to transmitter
        maxTxDist : TYPE, optional
            maximum distance to transmitter
        every : int
            use only every n-th receiver
        """
        # part 1: frequency axis
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

        # part 2: receiver axis
        if nInd is None:
            if minTxDist is not None or maxTxDist is not None:
                dTx = self.txDistance()
                minTxDist = minTxDist or 0
                maxTxDist = maxTxDist or 9e9
                nInd = np.nonzero((dTx >= minTxDist) * (dTx <= maxTxDist))[0]
            else:
                nInd = np.arange(len(self.rx))

            if isinstance(every, int):
                nInd = nInd[::every]

        if nInd is not None:
            for tok in ['alt', 'rx', 'ry', 'rz', 'line']:
                setattr(self, tok, getattr(self, tok)[nInd])

            self.DATA = self.DATA[:, :, nInd]
            if np.any(self.ERR):
                self.ERR = self.ERR[:, :, nInd]
            if np.any(self.RESP):
                self.RESP = self.RESP[:, :, nInd]
            if np.any(self.prim):
                for i in range(3):
                    self.prim[i] = self.prim[i][:, nInd]
            if hasattr(self, 'MODELS'):
                self.MODELS = self.MODELS[nInd, :]
            if self.prim is not None:
                for i in range(3):
                    self.prim[i] = self.prim[i][:, nInd]

        self.chooseData("data")  # make sure DATAX/Y/Z have correct size

    def mask(self, **kwargs):
        """Masking out data according to several properties."""
        pass  # not yet implemented

    def createConfig(self):
        """Create EMPYMOD input argument configuration."""
        self.cfg = {'src':
                    # [self.tx[0], self.tx[1], self.ty[0], self.ty[1], 0.1, 0.1],
                    [self.tx[0], self.tx[-1], self.ty[0], self.ty[-1], -0.1, -0.1],
                    # 'rec': [self.rx[0], self.ry[0], -self.alt[0], 0, 90],
                    'rec': [self.rx[0], self.ry[0], self.alt[0], 0, 90],
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

        self.cfg["rec"][:3] = self.rx[nrx], self.ry[nrx], self.alt[nrx]
        self.dataX = self.DATAX[:, nrx]
        self.dataY = self.DATAY[:, nrx]
        self.dataZ = self.DATAZ[:, nrx]
        self.nrx = nrx
        if show:
            self.showPos()

    def showPos(self, ax=None, line=None, background=None, org=False,
                color=None, marker=None):
        """Show positions."""
        if ax is None:
            fig, ax = plt.subplots()

        if org:
            # rxy = np.column_stack((self.rx, self.ry))
            pass

        ma = marker or "."
        ax.plot(self.rx, self.ry, ma, markersize=2, color=color or "blue")
        ax.plot(self.tx, self.ty, "-", markersize=4, color=color or "orange")
        if hasattr(self, "nrx") and self.nrx < self.nRx:
            ax.plot(self.rx[self.nrx], self.ry[self.nrx], "ko", markersize=5)

        if line is not None:
            ax.plot(self.rx[self.line == line],
                    self.ry[self.line == line], "-")

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
            transData = pg.trans.TransSymLog(tol=absError)
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
                ax = showSounding(data, self.f, ax=ax, color="C"+str(i), ls="",
                                  marker="x", label="B"+allcmp[i], **kwargs)
                if response is not None:
                    ax[0].plot(respRe[ncmp], self.f, ls="-", color="C"+str(i))
                    ax[1].plot(respIm[ncmp], self.f, ls="-", color="C"+str(i))
                    ncmp += 1

        for a in ax:
            a.legend()

        return ax

    def chooseData(self, what="data", llthres=None):
        """Choose data to show by showData or showLineData.

        Parameters
        ----------
        what : str
            property name or matrix to choose / show
                data - measured data
                prim - primary fields
                secdata - measured secondary data divided by primary fields
                response - forward response (NOT READY)
                error - absolute data error
                relerror - relative data error
                misfit - absolute misfit between data and response
                rmisfit - relative misfit between data and response
                wmisfit - error-weighted misfit
        """
        llthres = llthres or self.llthres
        if what.lower() == "data":
            self.DATAX, self.DATAY, self.DATAZ = self.DATA
        elif what.lower() == "prim":
            if self.prim is None:
                self.computePrimaryFields()

            self.DATAX, self.DATAY, self.DATAZ = self.prim
        elif what.lower() == "secdata":
            if self.prim is None:
                self.computePrimaryFields()

            primabs = np.sqrt(np.sum(self.prim**2, axis=0))
            self.DATAX, self.DATAY, self.DATAZ = self.DATA / primabs
        elif what.lower() == "response":
            self.DATAX, self.DATAY, self.DATAZ = self.RESP
        elif what.lower() == "error":
            self.DATAX, self.DATAY, self.DATAZ = self.ERR
        elif what.lower() == "relerror":
            rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-12)
            ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-12)
            self.DATAX, self.DATAY, self.DATAZ = rr + ii * 1j
        elif what.lower() == "misfit":
            self.DATAX, self.DATAY, self.DATAZ = self.DATA - self.RESP
            # [self.DATAX, self.DATAY, self.DATAZ] = [
            #     self.DATA[i] - self.RESP[i] for i in range(3)]
        elif what.lower() == "rmisfit":
            # for i, DD in enumerate([self.DATAX, self.DATAY, self.DATAZ]):
            for i in range(3):
                rr = (1. - self.RESP[i].real / self.DATA[i].real) * 100.
                ii = (1. - self.RESP[i].imag / self.DATA[i].imag) * 100.
                # rr[np.abs(rr) < llthres] = 0
                # ii[np.abs(ii) < llthres] = 0
                # DD = rr + ii *1j
                if i == 0:
                    self.DATAX = rr + ii * 1j
                elif i == 1:
                    self.DATAY = rr + ii * 1j
                elif i == 2:
                    self.DATAZ = rr + ii * 1j
        elif what.lower() == "wmisfit":
            mis = self.DATA - self.RESP
            wmis = mis.real / self.ERR.real + mis.imag / self.ERR.imag * 1j
            self.DATAX, self.DATAY, self.DATAZ = wmis
        else:  # try using the argument
            self.DATAX, self.DATAY, self.DATAZ = what

    def getWmisfit(self):

        mis = self.DATA - self.RESP
        self.WMIS = mis.real / self.ERR.real + mis.imag / self.ERR.imag * 1j

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
        log: bool [False]
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
        kwargs.setdefault("log", False)
        kwargs.setdefault("cmap", "jet")
        background = kwargs.pop("background", None)
        ax.plot(self.rx, self.ry, "k.", ms=1, zorder=-10)
        ax.plot(self.tx, self.ty, "k*-", zorder=-1)
        if isinstance(field, str):
            kwargs.setdefault("label", field)
            field = getattr(self, field)

        kwargs.setdefault("alim", [np.min(np.unique(field)),
                                   np.max(np.unique(field))])

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

    def showLineFreq(self, line=None, nf=0, ax=None, **kwargs):
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
            use logarithmic scale
        """

        kw = updatePlotKwargs(self.cmp, **kwargs)
        label = kwargs.pop("label", kw["what"])
        lw = kwargs.pop("lw", 0.5)
        self.chooseData(kw["what"], kw["llthres"])
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

        errbar = None
        if kw["what"] == 'data' and np.any(self.ERR):
            errbar = self.ERR[:, nf, nn]

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(kw["cmp"]), nrows=2,
                                   squeeze=False, sharex=True,
                                   sharey=not kw["amphi"],
                                   figsize=kwargs.pop("figsize", (10, 6)))
        else:
            fig = ax.flat[0].figure

        ncmp = 0
        allcmp = ['x', 'y', 'z']

        kwargs.setdefault("x", "x")
        if kwargs["x"] == "x":
            x = self.rx[nn]
        elif kwargs["x"] == "y":
            x = self.ry[nn]
        elif kwargs["x"] == "d":
            # need to eval line direction first, otherwise bugged
            # x = np.sqrt((self.rx[nn]-self.rx[0])**2+
            #             (self.ry[nn]-self.ry[0])**2)
            x = np.sqrt((np.mean(self.tx)-self.rx[nn])**2+
                        (np.mean(self.ty)-self.ry[nn])**2)

        for i in range(3):
            if kw["cmp"][i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[nf, nn]
                if kw["amphi"]:  # amplitude and phase
                    ax[0, ncmp].plot(x, np.abs(data), label=label)
                    ax[1, ncmp].plot(x, np.angle(data, deg=True), label=label)
                    if kw["log"]:
                        ax[0, ncmp].set_yscale('log')
                        ax[0, ncmp].set_ylim(kw["alim"])
                        ax[1, ncmp].set_ylim(kw["plim"])
                else:  # real and imaginary part
                    if kw["what"] == 'data' and errbar is not None:
                        ax[0, ncmp].errorbar(
                            x, np.real(data),
                            yerr=[errbar[i].real, errbar[i].real],
                            marker='o', lw=0., barsabove=True,
                            elinewidth=0.5, markersize=3, label=label)
                        ax[1, ncmp].errorbar(
                            x, np.imag(data),
                            yerr=[errbar[i].imag, errbar[i].imag],
                            marker='o', lw=0., barsabove=True,
                            elinewidth=0.5, markersize=3, label=label)
                    else:
                        ax[0, ncmp].plot(x, np.real(data), '+-', lw=lw,
                                               label=label)
                        ax[1, ncmp].plot(x, np.imag(data), '+-', lw=lw,
                                               label=label)
                    if kw["log"]:
                        ax[0, ncmp].set_yscale('symlog', linthresh=kw["llthres"])
                        ax[0, ncmp].set_ylim([-kw["alim"][1], kw["alim"][1]])
                        ax[1, ncmp].set_yscale('symlog', linthresh=kw["llthres"])
                        ax[1, ncmp].set_ylim([-kw["alim"][1], kw["alim"][1]])
                    else:
                        pass

                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1
        if kw["amphi"]:
            ax[0, 0].set_ylabel("Amplitude (nT/A)")
            ax[1, 0].set_ylabel("Phase (°)")
        else:
            ax[0, 0].set_ylabel("Real T (nT/A)")
            ax[1, 0].set_ylabel("Imag T (nT/A)")

        # if "x" not in kwargs:
        #     xt = np.round(np.linspace(0, len(nn)-1, 7))
        #     xtl = ["{:.0f}".format(self.rx[nn[int(xx)]]) for xx in xt]
        #     for aa in ax[-1, :]:
        #         if xtl[0] > xtl[-1]:
        #             aa.set_xlim([xt[1], xt[0]])

        #         aa.set_xticks(xt)
        #         aa.set_xticklabels(xtl)
        #         aa.set_xlabel("x (m)")

        for a in ax[-1, :]:
            if kwargs["x"] == "x":
                a.set_xlim([np.min(self.rx), np.max(self.rx)])
            elif kwargs["x"] == "y":
                a.set_xlim([np.min(self.ry), np.max(self.ry)])
            elif kwargs["x"] == "d":
                print('need to adjust xlim for *x* = *d* option')
            a.set_xlabel("x (m)")

        for a in ax.flat:
            a.set_aspect('auto')
            a.grid(True)

        if "what" in kwargs:
            self.chooseData("data", kw["llthres"])

        # plt.legend(["data", "response"])
        ax.flat[0].legend()

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)

        return fig, ax

    def showLineData(self, line=None, ax=None, **kwargs):
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
            use logarithmic scale
        """
        kw = updatePlotKwargs(self.cmp, **kwargs)
        self.chooseData(kw["what"], kw["llthres"])
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(kw["cmp"]), nrows=2,
                                   squeeze=False, sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (10, 6)))
        else:
            fig = ax.flat[0].figure

        ncmp = 0
        allcmp = ['x', 'y', 'z']

        for i in range(3):
            if kw["cmp"][i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[:, nn]
                print(data.shape)
                if kw["amphi"]:  # amplitude and phase
                    # pc1 = ax[0, ncmp].matshow(np.log10(np.abs(data)),
                    #                           cmap="Spectral_r")
                    norm = LogNorm(vmin=kw["alim"][0], vmax=kw["alim"][1])
                    pc1 = ax[0, ncmp].matshow(np.abs(data), norm=norm,
                                              cmap="Spectral_r")
                    if kw["alim"] is not None:
                        pc1.set_clim(kw["alim"])
                    pc2 = ax[1, ncmp].matshow(np.angle(data, deg=True),
                                              cmap="hsv")
                    pc2.set_clim(kw["plim"])
                else:  # real and imaginary part
                    if kw["log"]:
                        pc1 = ax[0, ncmp].matshow(
                            np.real(data),
                            norm=SymLogNorm(linthresh=kw["alim"][0],
                                            vmin=-kw["alim"][1],
                                            vmax=kw["alim"][1]),
                            cmap=kw["cmap"])
                        pc2 = ax[1, ncmp].matshow(
                            np.imag(data),
                            norm=SymLogNorm(linthresh=kw["alim"][0],
                                            vmin=-kw["alim"][1],
                                            vmax=kw["alim"][1]),
                            cmap=kw["cmap"])
                    else:
                        pc1 = ax[0, ncmp].matshow(np.real(data),
                                                  cmap=kw["cmap"])
                        if kw["alim"] is not None:
                            pc1.set_clim([kw["alim"][0], kw["alim"][1]])
                        pc2 = ax[1, ncmp].matshow(np.imag(data),
                                                  cmap=kw["cmap"])
                        if kw["alim"] is not None:
                            pc2.set_clim([kw["alim"][0], kw["alim"][1]])

                for j, pc in enumerate([pc1, pc2]):
                    divider = make_axes_locatable(ax[j, ncmp])
                    cax = divider.append_axes("right", size="5%", pad=0.15)
                    cb = plt.colorbar(pc, cax=cax, orientation="vertical")
                    if not kw["amphi"]:
                        if ncmp + 1 == sum(kw["cmp"]) and kw["log"]:
                            makeSymlogTicks(cb, kw["alim"])
                        elif ncmp + 1 == sum(kw["cmp"]) and not kw["log"]:
                            pass
                        else:
                            cb.set_ticks([])
                    else:
                        tit = "log10(B) in nT/A" if j == 0 else "phi in °"
                        cb.ax.set_title(tit)

                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1

        ax[0, 0].set_ylim([-0.5, len(self.f)-0.5])
        yt = np.arange(0, len(self.f), 2)
        ytl = ["{:.0f}".format(self.f[yy]) for yy in yt]
        for aa in ax[:, 0]:
            aa.set_yticks(yt)
            aa.set_yticklabels(ytl)
            aa.set_ylabel("f (Hz)")

        xt = np.round(np.linspace(0, len(nn)-1, 7))
        xtl = ["{:.0f}".format(self.rx[nn[int(xx)]]) for xx in xt]

        for aa in ax[-1, :]:
            if xtl[0] > xtl[-1]:
                aa.set_xlim([xt[1], xt[0]])
            aa.set_xticks(xt)
            aa.set_xticklabels(xtl)
            aa.set_xlabel("x (m)")

        for a in ax.flat:
            a.set_aspect('auto')

        if "what" in kwargs:
            self.chooseData("data", kw["llthres"])

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)

        return fig, ax

    def showPatchData(self, nf=0, ax=None, figsize=(12, 6),
                      scale=0, background=None, **kwargs):
        """Show all three components as amp/phi or real/imag plots.

        Parameters
        ----------
        nf : int | float
            frequency index (int) or value (float) to plot
        """
        kw = updatePlotKwargs(self.cmp, **kwargs)
        kw.setdefault("numpoints", 0)
        kw.setdefault("radius", self.radius)
        self.chooseData(kw["what"], kw["llthres"])
        amap = dMap("Spectral")  # mirrored

        if isinstance(nf, float):
            nf = np.argmin(np.abs(self.f - nf))
            if self.verbose:
                print("Chose no f({:d})={:.0f} Hz".format(nf, self.f[nf]))

        if background is not None and kwargs.pop("overlay", False):  # bwc
            background = "BKG"

        allcmp = np.take(["x", "y", "z"], np.nonzero(kw["cmp"])[0])
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
            if kw["amphi"]:
                kw.pop("cmap", None)
                alim = kw.pop("alim", [1e-3, 1])
                plim = kw.pop("plim", [-180, 180])
                kw.pop("log", None)
                plotSymbols(self.rx, self.ry, np.abs(data[nf]),
                            ax=ax[0, j], colorBar=(j == len(allcmp)-1), **kw,
                            cmap=amap, log=True, alim=alim)
                plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                            ax=ax[1, j], colorBar=(j == len(allcmp)-1), **kw,
                            cmap="hsv", log=False, alim=plim)
                ax[0, j].set_title("log10 T"+cc+" [pT/A]")
                ax[1, j].set_title(r"$\phi$"+cc+" [°]")
            else:
                _, cb1 = plotSymbols(self.rx, self.ry, np.real(data[nf]),
                                     ax=ax[0, j], **kw)
                _, cb2 = plotSymbols(self.rx, self.ry, np.imag(data[nf]),
                                     ax=ax[1, j], **kw)

                ax[0, j].set_title("real T"+cc+" [nT/A]")
                ax[1, j].set_title("imag T"+cc+" [nT/A]")

                for cb in [cb1, cb2]:
                    if ncmp + 1 == sum(kw["cmp"]) and kw["log"]:
                        makeSymlogTicks(cb, kw["alim"])
                    elif ncmp + 1 == sum(kw["cmp"]) and not kw["log"]:
                        pass
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
            basename += " " + kwargs["what"]

        fig.suptitle(basename+"  f="+str(self.f[nf])+"Hz")

        if "what" in kwargs:
            self.chooseData("data", kw["llthres"])

        return fig, ax

    def showLineData2(self, line=None, ax=None, **kwargs):
        """Show alternative line plot.
        """

        kw = updatePlotKwargs(self.cmp, **kwargs)
        self.chooseData(kw["what"], kw["llthres"])
        kw.setdefault("radius", "rect")
        if 'x' in kw:
            kwx = kw.pop('x')
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(kw["cmp"]), nrows=2,
                                   squeeze=False, sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (10, 6)))
        else:
            fig = ax.flat[0].figure

        ncmp = 0
        allcmp = ['x', 'y', 'z']

        if kwx == "x":
            x = np.tile(self.rx[nn], len(self.f))
        elif kwx == "y":
            x = np.tile(self.ry[nn], len(self.f))
        elif kwx == "d":
            print('need to implement *d* option')
            # need to eval line direction first, otherwise bugged
            # x = np.sqrt((self.rx[nn]-self.rx[0])**2+
            #             (self.ry[nn]-self.ry[0])**2)
            # x = np.sqrt((np.mean(self.tx)-self.rx[nn])**2+
            #             (np.mean(self.ty)-self.ry[nn])**2)
        y = np.repeat(np.arange(len(self.f)), len(nn))

        for i in range(3):
            if kw["cmp"][i] > 0:
                data = getattr(self, "DATA"+allcmp[i].upper())[:, nn].ravel()
                if kw["amphi"]:
                    print('need to implement amphi')
                    # kwargs.pop("cmap", None)
                    # kwargs.pop("log", None)
                    # kwargs.pop("alim", None)
                    # plotSymbols(self.rx, self.ry, np.abs(data[nf]),
                    #             ax=ax[0, j], colorBar=(j == len(allcmp)-1), **kw,
                    #             cmap="Spectral_r", log=True, alim=kw["alim"])
                    # plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                    #             ax=ax[1, j], colorBar=(j == len(allcmp)-1), **kw,
                    #             cmap="hsv", log=False, alim=kw["plim"])
                    # ax[0, j].set_title("log10 T"+cc+" [pT/A]")
                    # ax[1, j].set_title(r"$\phi$"+cc+" [°]")
                else:
                    _, cb1 = plotSymbols(x, y, np.real(data),
                                         ax=ax[0, ncmp], **kw)
                    _, cb2 = plotSymbols(x, y, np.imag(data),
                                         ax=ax[1, ncmp], **kw)

                    for cb in [cb1, cb2]:
                        if ncmp + 1 == sum(kw["cmp"]) and kw["log"]:
                            makeSymlogTicks(cb, kw["alim"])
                        elif ncmp + 1 == sum(kw["cmp"]) and not kw["log"]:
                            pass
                        else:
                            cb.set_ticks([])
                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1

        ax[0, 0].set_ylim([0., len(self.f)])
        for a in ax[-1, :]:
            if kwargs["x"] == "x":
                a.set_xlim([np.min(self.rx), np.max(self.rx)])
            elif kwargs["x"] == "y":
                a.set_xlim([np.min(self.ry), np.max(self.ry)])
            elif kwargs["x"] == "d":
                print('need to adjust xlim for *x* = *d* option')
            a.set_xlabel("x (m)")
        yt = np.arange(0, len(self.f), 2)
        ytl = ["{:.0f}".format(self.f[yy]) for yy in yt]
        for aa in ax[:, 0]:
            aa.set_yticks(yt)
            aa.set_yticklabels(ytl)
            aa.set_ylabel("f (Hz)")

        for a in ax.flat:
            a.set_aspect('auto')

        if "what" in kwargs:
            self.chooseData("data", kw["llthres"])

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)

        return fig, ax

    def showData(self, *args, **kwargs):
        """Generic show function.

        Upon keyword arguments given, directs to
        * showDataPatch [default]
        * showLineData (if line=given)
        * showLineFreq (if line and nf given)
        """
        if "line" in kwargs:
            if "nf" in kwargs:
                return self.showLineFreq(*args, **kwargs)
            else:
                return self.showLineData2(*args, **kwargs)
        else:
            return self.showPatchData(*args, **kwargs)

    def computePrimaryFields(self):
        """Compute primary fields."""
        cfg = dict(self.cfg)
        fak = 4e-7 * np.pi * 1e9  # H->B and T in nT
        cfg["rec"] = [self.rx, self.ry, self.alt, 0, 0]  # x
        cfg["freqtime"] = self.f
        cfg["xdirect"] = True
        cfg["res"] = [2e14]
        cfg["depth"] = []
        print(cfg)
        pfx = bipole(**cfg).real * fak
        cfg["rec"][3:5] = [90, 0]  # y
        pfy = bipole(**cfg).real * fak
        cfg["rec"][3:5] = [0, 90]  # z
        pfz = bipole(**cfg).real * fak
        self.prim = np.stack([pfx, pfy, pfz])

    def generateDataPDF(self, pdffile=None, figsize=[12, 6],
                        mode='patchwise', **kwargs):
        """Generate a multi-page pdf file containing all data."""
        what = kwargs.setdefault('what', 'data')
        llthres = kwargs.pop('llthres', self.llthres)
        self.chooseData(what, llthres)

        if mode == 'patchwise':
            pdffile = pdffile or self.basename + "-" + what + ".pdf"
        elif mode == 'linewise':
            pdffile = pdffile or self.basename + "-line-" + what + ".pdf"
        elif mode == 'linewise2':
            pdffile = pdffile or self.basename + "-line-" + what + ".pdf"
        elif mode == 'linefreqwise':
            pdffile = pdffile or self.basename + "-linefreqs-" + what + ".pdf"
        else:
            print('Error, wrong *mode* chosen. Aborting ...')
            raise SystemExit

        with PdfPages(pdffile) as pdf:
            if mode == 'linefreqwise':
                fig, ax = plt.subplots(figsize=figsize)
                self.showField(self.line, ax=ax)
                ax.figure.savefig(pdf, format="pdf")

                ul = np.unique(self.line)
                plt.close(fig)
                for li in ul[ul > 0]:
                    nn = np.nonzero(self.line == li)[0]
                    if np.isfinite(li) and len(nn) > 3:
                        for fi, freq in enumerate(self.f):
                            kwargs["what"] = "data"
                            fig, ax = self.showLineFreq(li, fi,
                                                        **kwargs)
                            kwargs["what"] = "response"
                            fig, ax = self.showLineFreq(li, fi, ax=ax,
                                                        **kwargs)
                            fig.suptitle('line = {:.0f}, '
                                         'freq = {:.0f} Hz'.format(li, freq))
                            fig.savefig(pdf, format='pdf')
                            plt.close(fig)

            elif mode == 'linewise':
                fig, ax = plt.subplots(figsize=figsize)
                self.showField(self.line, ax=ax)
                ax.figure.savefig(pdf, format="pdf")

                ul = np.unique(self.line)
                plt.close(fig)
                for li in ul[ul > 0]:
                    nn = np.nonzero(self.line == li)[0]
                    if np.isfinite(li) and len(nn) > 3:
                        fig, ax = self.showLineData(li, **kwargs)
                        fig.suptitle('line = {:.0f}'.format(li))
                        fig.savefig(pdf, format='pdf')
                        plt.close(fig)
            elif mode == 'linewise2':
                fig, ax = plt.subplots(figsize=figsize)
                self.showField(self.line, ax=ax)
                ax.figure.savefig(pdf, format="pdf")

                ul = np.unique(self.line)
                plt.close(fig)
                for li in ul[ul > 0]:
                    nn = np.nonzero(self.line == li)[0]
                    if np.isfinite(li) and len(nn) > 3:
                        fig, ax = self.showLineData2(li, **kwargs)
                        fig.suptitle('line = {:.0f}'.format(li))
                        fig.savefig(pdf, format='pdf')
                        plt.close(fig)
            else:
                fig, ax = plt.subplots(ncols=2, figsize=figsize, sharey=True)
                self.showField(np.arange(len(self.rx)), ax=ax[0],
                               cMap="Spectral_r")
                ax[0].set_title("Sounding number")
                self.showField(self.line, ax=ax[1], cMap="Spectral_r")
                ax[1].set_title("Line number")
                fig.savefig(pdf, format='pdf')
                plt.close(fig)
                for i in range(len(self.f)):
                    fig, ax = self.showPatchData(nf=i, figsize=figsize,
                                            **kwargs)
                    fig.savefig(pdf, format='pdf')
                    plt.close(fig)

    def generateModelPDF(self, pdffile=None, **kwargs):
        """Generate a PDF of all models."""
        dep = self.depth.copy()
        dep[:-1] += np.diff(self.depth) / 2
        pdffile = pdffile or self.basename + "-models5.pdf"
        kwargs.setdefault('alim', [5, 5000])
        kwargs.setdefault('log', True)
        with PdfPages(pdffile) as pdf:
            fig, ax = plt.subplots()
            for i in range(self.allModels.shape[1]):
                self.showField(self.allModels[:, i], ax=ax, **kwargs)
                ax.set_title('z = {:.1f}'.format(dep[i]))
                fig.savefig(pdf, format='pdf')  # , bbox_inches="tight")
                ax.cla()

    def estimateError(self, ignoreErr=True, useMax=False, **kwargs):
        """Estimate data error to be saved in self.ERR.

        Errors can be
        A) a sum of absolute and relative error (and the processing error)
        B) the maximum of all contributions (relative, absolute, processing)

        Parameters
        ----------
        relError

        absError
        """

        absError = kwargs.pop("absError", self.llthres)
        relError = kwargs.pop("relError", 0.05)
        aErr = np.zeros_like(self.DATA, dtype=complex)
        aErr.real = absError
        aErr.imag = absError
        rErr = np.abs(self.DATA.real) * relError + \
            np.abs(self.DATA.imag) * relError * 1j

        if ignoreErr:
            self.ERR = np.zeros_like(self.DATA)

        # decide upon adding or maximizing errors
        if useMax:
            self.ERR = np.maximum(np.maximum(self.ERR, aErr), rErr)
        else:
            self.ERR = self.ERR + aErr + rErr

    def deactivateNoisyData(self, aErr=None, rErr=None):
        """Set data below a certain threshold to nan (inactive)."""

        if aErr is not None:
            self.DATA[np.abs(self.DATA) < aErr] = np.nan + 1j * np.nan
            self.DATA[np.abs(self.DATA) < aErr] = np.nan + 1j * np.nan

        if rErr is not None:
            rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-12)
            ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-12)

            self.DATA[np.abs(rr) > rErr] = np.nan + 1j * np.nan
            self.DATA[np.abs(ii) > rErr] = np.nan + 1j * np.nan

    def getData(self, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.pop("cmp", self.cmp)
        if np.shape(self.ERR) != np.shape(self.DATA):
            self.estimateError(**kwargs)

        if line is None:  # take all existing (nonzero) lines
            ind = np.nonzero(self.line > 0)[0]
        else:
            ind = np.nonzero(self.line == line)[0]

        allcmp = ['X', 'Y', 'Z']
        meany = 0  # np.median(self.ry[ind]) # needed anymore?
        ypos = np.round((self.ry[ind]-meany)*10)/10  # get to straight line
        rxpos = np.round(np.column_stack((self.rx[ind], ypos,
                                          self.rz[ind]-self.txAlt))*10)/10
        nF = len(self.f)
        nT = 1
        nR = len(ind)  # rxpos.shape[0]
        nC = sum(cmp)
        dataR = np.zeros([nT, nC, nF, nR])
        dataI = np.zeros_like(dataR)
        errorR = np.zeros_like(dataR)
        errorI = np.zeros_like(dataR)
        kC = 0
        Cmp = []
        for iC in range(3):
            if cmp[iC]:
                dataR[0, kC, :, :] = self.DATA[iC][:, ind].real
                dataI[0, kC, :, :] = self.DATA[iC][:, ind].imag
                errorR[0, kC, :, :] = self.ERR[iC][:, ind].real
                errorI[0, kC, :, :] = self.ERR[iC][:, ind].imag
                Cmp.append('B'+allcmp[iC].lower())
                kC += 1

        # error estimation
        data = dict(dataR=dataR, dataI=dataI,
                    errorR=errorR, errorI=errorI,
                    rx=rxpos, cmp=Cmp)

        return data

    def saveData(self, fname=None, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.setdefault("cmp", self.cmp)
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

        if line == "all":
            line = np.arange(1, max(self.line)+1)

        if hasattr(line, "__iter__"):
            for i in line:
                self.saveData(line=i)
            return

        data = self.getData(line=line, **kwargs)
        data["tx_ids"]=[0]
        DATA = [data]
        meany = 0  # np.median(self.ry[ind]) # needed anymore?
        np.savez(fname+".npz",
                 tx=[np.column_stack((self.tx, self.ty-meany, self.tx*0))],
                 freqs=self.f,
                 cmp=cmp,
                 DATA=DATA,
                 line=self.line,
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
        self.loadResponse(dirname)
        self.J = None
        if os.path.exists(dirname+"invmesh.vtk"):
            self.mesh = pg.load(dirname+"invmesh.vtk")
        else:
            self.mesh = pg.load(dirname + datafile + "_final_invmodel.vtk")

        jacobian = jacobian or datafile+"_jacobian.bmat"
        jname = dirname + jacobian
        if os.path.exists(jname):
            self.J = pg.load(jname)
            print("Loaded jacobian: "+jname, self.J.rows(), self.J.cols())
        elif os.path.exists(dirname+"jacobian.bmat"):
            self.J = pg.load(dirname+"jacobian.bmat")
            print("Loaded jacobian: ", self.J.rows(), self.J.cols())

    def getIndices(self):
        """."""
        ff = np.array([], dtype=bool)
        for i in range(3):
            if self.cmp[i]:
                tmp = self.DATA[i].ravel() * self.ERR[i].ravel()
                ff = np.hstack((ff, np.isfinite(tmp)))

        return ff

    def nData(self):
        """Number of data (for splitting the response)."""
        return sum(self.getIndices())

    def loadResponse(self, dirname=None, response=None):
        """Load model response file."""
        if response is None:
            respfiles = sorted(glob(dirname+"response_iter*.npy"))
            if len(respfiles) == 0:
                respfiles = sorted(glob(dirname+"reponse_iter*.npy"))  # TYPO
            if len(respfiles) == 0:
                pg.error("Could not find response file")

            responseVec = np.load(respfiles[-1])
            respR, respI = np.split(responseVec, 2)
            response = respR + respI*1j


        sizes = [sum(self.cmp), self.nF, self.nRx]
        RESP = np.ones(np.prod(sizes), dtype=np.complex) * np.nan
        try:
            RESP[self.getIndices()] = response
        except ValueError:
            RESP[:] = response

        RESP = np.reshape(RESP, sizes)
        self.RESP = np.ones((3, self.nF, self.nRx), dtype=np.complex) * np.nan
        self.RESP[np.nonzero(self.cmp)[0]] = RESP

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
