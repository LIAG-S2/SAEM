"""Controlled-source electromagnetic (CSEM) data class."""
from glob import glob
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.viewer.mpl import drawModel1D
from pygimli.viewer.mpl import showStitchedModels
from pygimli.core.math import symlog

from .plotting import showSounding
from .emdata import EMData
from .modelling import fopSAEM, bipole
from .tools import distToTx


class CSEMData(EMData):
    """Class for CSEM frequency-domain data patch (single Tx)."""

    def __init__(self, datafile=None, **kwargs):
        """Initialize CSEM data class.

        Parameters
        ----------
        datafile : str
            data file to load if not None
        basename : str [datafile without extension]
            name for data (exporting, figures etc.)
        mode : str ['B']
            measuring quantity ('E', 'B' or 'EB')
        txPos : array
            transmitter position as polygone
        rx/ry/rz : iterable
            receiver positions
        tx/ty/tz : iterable
            transmitter positions
        f : iterable
            frequencies
        cmp : [int, int, int]
            active components
        alt : float
            flight altitude
        """
        super().__init__(**kwargs)
        self.mode = kwargs.pop("mode", "B")
        self.createDataArray()
        self.loop = kwargs.pop("loop", False)
        self.txAlt = kwargs.pop("txalt", 0.0)
        self.alt = self.rz - self.txAlt

        if datafile is not None:
            self.loadData(datafile)

        if len(self.rx) > 1:
            dxy = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
            self.radius = np.median(dxy) * 0.5
            self.createConfig()

    def __repr__(self):
        """String representation of the class."""
        sdata = "CSEM data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))
        txlen = np.sum(np.sqrt(np.diff(self.tx)**2+np.diff(self.ty)**2))
        stx = "Transmitter length {:.0f}m".format(txlen)
        dx = np.sqrt(np.diff(self.rx)**2+np.diff(self.ry)**2)
        smrx = "Median Rx distance {:.1f}m".format(np.median(dx))
        spos = "Sounding pos at " + (3*"{:1f},").format(*self.cfg["rec"][:3])

        return "\n".join((sdata, stx, smrx, spos))

    def createDataArray(self):
        """Create data array for a given model ("E", "B", or "RB")."""
        if self.mode == 'EB':
            self.cstr = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
        elif self.mode == 'B':
            self.cstr = ['Bx', 'By', 'Bz']
        elif self.mode == 'E':
            self.cstr = ['Ex', 'Ey', 'Ez']
        else:
            print('Error! Choose correct mode for CSEMData initialization.')
            raise SystemExit
        self.DATA = np.zeros((len(self.cstr), self.nF, self.nRx),
                             dtype=complex)
        self.cmp = np.ones(len(self.cstr), dtype=bool)

    def loadData(self, filename, detectLines=False):
        """Load any data format."""
        if filename.endswith(".npz"):
            self.loadNpzFile(filename)
        elif filename.endswith(".mat"):
            if not self.loadEmteresMatFile(filename):
                print("No frequency in data, try read WWU style")
                self.loadWWUMatFile(filename)

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
            assert np.allclose(self.f, new.f, rtol=0.01), "frequencies differ!"
        for attr in ["rx", "ry", "rz", "line", "alt",
                     'DATA', 'ERR', 'RESP', 'PRIM']:
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
        txgeo = ALL["tx"][nr][:, :3].T
        data = ALL["DATA"][nr]
        rxs = np.array(data["rx"])
        self.__init__(txPos=txgeo, f=freqs,
                      rx=rxs[:, 0], ry=rxs[:, 1], rz=rxs[:, 2],
                      mode=self.mode)

        if 'line' in ALL:
            self.line = ALL["line"]

        self.DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        self.DATAY = np.zeros_like(self.DATAX)
        self.DATAZ = np.zeros_like(self.DATAX)
        try:
            cmp = ALL["DATA"][nr]["cmp"]
            for cstr in cmp:
                try:
                    idx = self.cstr.index(cstr)
                    self.cmp[idx] = 1
                except ValueError:
                    self.cmp[idx] = 0

        except Exception:
            print('CMP detect change exception, using old way')
            print(Exception)
            self.cmp = [np.any(getattr(self, "DATA"+cc))
                        for cc in ["X", "Y", "Z"]]

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

    def loadEmteresMatFile(self, filename):
        """Load data from mat file (WWU Muenster processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
        print(filenames)
        assert len(filenames) > 0
        filename = filenames[0]
        MAT = loadmat(filename)
        if "f" not in MAT:
            return False

        MAT["line"] = np.ones(MAT["lon"].shape[-1], dtype=int)
        line = 1
        if len(filenames) > 1:
            print("read "+filename)
        for filename in filenames[1:]:
            print("reading "+filename)
            MAT1 = loadmat(filename)
            line += 1
            MAT1["line"] = np.ones(MAT1["lon"].shape[-1], dtype=int) * line
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

    def loadWWUMatFile(self, filename):
        """Load data from mat file (WWU processing)."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
        assert len(filenames) > 0
        filename = filenames[0]
        # ALL = loadmat(filename)
        # if "zfts" in ALL:
        #     MAT = ALL["ztfs"][0][0]
        # elif "data" in ALL:
        #     MAT = ALL["data"][0][0]
        MAT = loadmat(filename)["ztfs"][0][0]
        MAT["line"] = np.ones(MAT["xy"].shape[-1], dtype=int)
        line = 1
        if len(filenames) > 1:
            print("read "+filename)
        for filename in filenames[1:]:
            print("reading "+filename)
            MAT1 = loadmat(filename)["ztfs"][0][0]
            line += 1
            MAT1["line"] = np.ones(MAT1["xy"].shape[-1], dtype=int) * line
            for name in MAT.dtype.names:
                if name == "periods":
                    assert len(MAT[name]) == len(MAT1[name]), "nFreqs no match"
                    assert np.allclose(MAT[name], MAT1[name]), "freqs no match"
                else:
                    MAT[name] = np.concatenate((MAT[name], MAT1[name]),
                                               axis=-1)
        self.MAT = MAT

        self.f = np.round(100.0 / np.squeeze(MAT["periods"])) / 100.
        self.ry, self.rx = MAT["xy"]

        # if "topo" in MAT.dtype.names:
        #     self.rz = MAT["topo"][0]
        if "lla" in MAT.dtype.names:
            self.rz = MAT["lla"][2]
        else:
            raise Exception("Could not determine altitude!")

        if "line" in MAT.dtype.names:
            self.line = MAT["line"]

        TMP = np.squeeze(MAT["tfs"])
        if len(TMP.shape) != 3:
            TMP2 = np.zeros((3, *TMP.shape), dtype=complex)
            TMP2[2, :, :] = TMP
            TMP = TMP2
            self.cmp = [0, 0, 1]

        DY, DX, DZ = TMP

        self.DATA = np.stack((-DX, -DY, DZ))
        TMP = np.squeeze(MAT["tfs_se"])
        if len(TMP.shape) != 3:
            TMP2 = np.zeros((3, *TMP.shape), dtype=complex)
            TMP2[2, :, :] = TMP
            TMP = TMP2

        if TMP.dtype is complex:
            self.ERR = TMP
        else:
            self.ERR = TMP * (1+1j)

        self.alt = self.rz - self.txAlt
        return True

    def simulate(self, rho, thk=[], **kwargs):
        """Simulate data by assuming 1D layered model.

        Parameters
        ----------
        rho : float|iterable
            Resistivity (vector)
        thk : []|float|iterable
            Thickness vector (len(thk)+1 must be len(rho))
        fullTx : bool [False]
            Compute full segments (slower)
        show : bool [False]
            Show result
        **kwargs are passed to the show function
        """
        cmp = [1, 1, 1]  # cmp = kwargs.pop("cmp", self.cmp)
        self.createConfig(fullTx=kwargs.pop("fullTx", False))
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
            self.showData(what="response", **kwargs)

    def computePrimaryFields(self):
        """Compute primary fields."""
        cfg = dict(self.cfg)
        fak = 4e-7 * np.pi * 1e9  # H->B and T in nT
        self.alt = self.rz - np.mean(self.tz)
        cfg["rec"] = [self.rx, self.ry, self.alt, 0, 0]  # x
        cfg["freqtime"] = self.f
        cfg["xdirect"] = True
        cfg["res"] = [2e14]
        cfg["depth"] = []
        pfx = bipole(**cfg).real * fak
        cfg["rec"][3:5] = [90, 0]  # y
        pfy = bipole(**cfg).real * fak
        cfg["rec"][3:5] = [0, 90]  # z
        pfz = bipole(**cfg).real * fak
        self.PRIM = np.stack([pfx, pfy, pfz])

    def txDistance(self, seg=True):
        """Distance to transmitter."""
        if seg:  # segment-wise
            return distToTx(self.rx, self.ry, self.tx, self.ty)
        else:  # old: rotate and x distance
            ang = np.median(np.arctan2(np.diff(self.ty), np.diff(self.tx)))
            ang += np.pi / 2
            A = np.array([[np.cos(ang), np.sin(ang)],
                          [-np.sin(ang), np.cos(ang)]])
            rx, ry = A.dot(np.array([self.rx-np.mean(self.tx),
                                     self.ry-np.mean(self.ty)]))
            return np.abs(rx)

    def switchTx(self):
        """Switch orientation of transmitter."""
        self.tx = self.tx[::-1]
        self.ty = self.ty[::-1]
        self.tz = self.tz[::-1]

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
        """Show amplitude and phase data.

        Parameters
        ----------
        nrx : int
            receiver number
        position : [float, float]
            position to search for the next receiver
        ax : [Axes, Axes]
            two matplotlib axes objects to plot into (otherwise new)


        Returns
        -------
        ax : [Axes, Axes]
            two matplotlib axes objects
        """
        cmp = kwargs.pop("cmp", self.cmp)
        if nrx is not None or position is not None:
            self.setPos(nrx, position)

        ax = kwargs.pop("ax", None)
        allcmp = ['x', 'y', 'z']
        if response is not None:
            if response is True:
                respRe = np.stack([self.RESP[i, :, self.nrx].real
                                   for i in np.nonzero(cmp)[0]])
                respIm = np.stack([self.RESP[i, :, self.nrx].imag
                                   for i in np.nonzero(cmp)[0]])
            else:
                respRe, respIm = np.reshape(response, (2, -1))
                respRe = np.reshape(respRe, (sum(cmp), -1))
                respIm = np.reshape(respIm, (sum(cmp), -1))

        ncmp = 0
        amphi = kwargs.pop("amphi", True)
        for i in range(3):
            if cmp[i] > 0:
                data = getattr(self, "data"+allcmp[i].upper())
                kwargs.setdefault("color", "C" + str(i))
                kwargs.setdefault("label", "B" + allcmp[i])
                ax = showSounding(data, self.f, ax=ax, ls="",
                                  marker="x", amphi=amphi, **kwargs)
                if response is not None:
                    # col = kwargs["color"]
                    if amphi:
                        snddata = respRe[ncmp] + respIm[ncmp] * 1j
                        ax[0].plot(np.abs(snddata), self.f, ls="-", **kwargs)
                        ax[1].plot(np.angle(snddata)*180/np.pi, self.f, ls="-", **kwargs)
                    else:
                        ax[0].plot(respRe[ncmp], self.f, ls="-", **kwargs)
                        ax[1].plot(respIm[ncmp], self.f, ls="-", **kwargs)

            ncmp += 1

        for a in ax:
            a.legend()

        return ax

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
    data = CSEMData(datafile="data_f*.mat", txPos=txpos, txalt=70)
    print(data)
    # self.generateDataPDF()
    data.showData(nf=1)
    # self.showField("alt", background="BKG")
    # self.invertSounding(nrx=20)
    # plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
    data.showSounding(nrx=20)
    # self.showData(nf=1)
    # self.generateDataPDF()
