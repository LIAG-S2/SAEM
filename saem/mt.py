"""Magnetotellurics (MT) data class, derived from EMData."""
from glob import glob
import os.path
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D
from pygimli.viewer.mpl import showStitchedModels
from pygimli.core.math import symlog
# from matplotlib.colors import LogNorm, SymLogNorm

# from .plotting import plotSymbols, showSounding, updatePlotKwargs
from .plotting import showSounding
# from .plotting import underlayBackground, makeSymlogTicks, dMap
from .emdata import EMData
from .modelling import fopSAEM  # , bipole
# from .tools import readCoordsFromKML, distToTx, detectLinesAlongAxis
from .tools import distToTx
# from .tools import detectLinesBySpacing, detectLinesByDistance


class MTData(EMData):
    """Class for MT frequency-domain data patch."""

    def __init__(self, datafile=None, mode='ZT', **kwargs):
        """Initialize MT data class.

        Data, error, response array sizes might be reduced by specifying a
        specific mode.

        Parameters
        ----------
        datafile : str
            data file to load if not None

        mode : str
            *ZT* general MT/AFMAG data array (Zxx, Zxy, Zyx, Zyy, Tx, Ty)
            *Z* for full impedance tensor (Zxx, Zxy, Zyx, Zyy)
            *Zd* for diagonal of impedance tensor (Zxx, Zyy)
            *Zo* for off-diagonal of impedance tensor (Zxy, Zyx)
            *T* for tipper (Tx, Ty)
        """
        super().__init__()

        self.mode = mode
        self.tx = [[], []]
        self.ty = [[], []]
        self.tz = [[], []]
        self.updateDefaults(**kwargs)
        self.createDataArray()

        if 'debugImport' in kwargs:
            self.firstonly = True
        else:
            self.firstonly = False

        if datafile is not None:
            self.loadData(datafile)

        dxy = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        self.radius = np.median(dxy) * 0.5

        # self.createConfig()

    def __repr__(self):
        """String representation of the class."""
        sdata = "MT data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))

        return "\n".join((sdata))

    def createDataArray(self):
        """Create data arrays according to given mode (ZT, Z, T, Zd, or Zo)."""
        if self.mode == 'ZT':
            self.cstr = ['Zxx', 'Zxy', 'Zyx', 'Zyy', 'Tx', 'Ty']
        elif self.mode == 'Z':
            self.cstr = ['Zxx', 'Zxy', 'Zyx', 'Zyy']
        elif self.mode == 'T':
            self.cstr = ['Tx', 'Ty']
        elif self.mode == 'Zd':
            self.cstr = ['Zxx', 'Zyy']
        elif self.mode == 'Zo':
            self.cstr = ['Zxy', 'Zyx']
        else:
            print('Error! Choose correct mode for MTData initialization.')
            raise SystemExit
        self.DATA = np.zeros((len(self.cstr), self.nF, self.nRx),
                             dtype=complex)
        self.cmp = np.ones(len(self.cstr), dtype=bool)

    def loadData(self, filename, detectLines=False):
        """Load any data format."""
        if filename.endswith(".npz"):
            self.loadNpzFile(filename)
        elif filename.endswith(".mat"):
            self.loadIPHTMatFile(filename)

        if len(self.line) != len(self.rx):
            self.line = np.ones_like(self.rx, dtype=int)

        if detectLines:
            self.detectLines()

    def addData(self, new, detectLines=False):
        """Add (concatenate) data."""
        if isinstance(new, str):
            new = MTData(new)
        if new.tx is not None:
            assert np.allclose(self.tx, new.tx), "Tx(x) not matching!"
        if new.ty is not None:
            assert np.allclose(self.ty, new.ty), "Tx(y) not matching!"
        if new.f is not None:
            assert np.allclose(self.f, new.f)
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
        data = ALL["DATA"][nr]
        rxs = np.array(data["rx"])
        self.__init__(f=freqs, rx=rxs[:, 0], ry=rxs[:, 1], rz=rxs[:, 2],
                      mode=self.mode)

        if 'line' in ALL:
            self.line = ALL["line"]

        data = ALL["DATA"][nr]
        # do not overwrite cmp at this point until correct defined
        # self.cmp = ALL["cmp"]
        self.ACTIVE = np.zeros_like(self.DATA)
        self.ERR = np.zeros_like(self.DATA)

        for ic, cmp in enumerate(data["cmp"]):
            self.ACTIVE[ic, :] = data["errorR"][0, ic, :, :] + \
                                 data["errorI"][0, ic, :, :] * 1j
        self.ERR[:] = self.ACTIVE[:]
        for ic, cmp in enumerate(data["cmp"]):
            self.ACTIVE[ic, :] = data["dataR"][0, ic, :, :] + \
                                 data["dataI"][0, ic, :, :] * 1j
        self.DATA[:] = self.ACTIVE[:]

    def loadIPHTMatFile(self, filename):
        """Load IPHT style Matlab data file."""
        self.basename = filename.replace("*", "").replace(".mat", "")
        filenames = sorted(glob(filename))
        assert len(filenames) > 0
        filename = filenames[0]
        MAT = loadmat(filename)
        if len([var for var in MAT.keys() if "_" not in var]) == 1:
            MAT = MAT[[var for var in MAT.keys() if "_" not in var][0]][0]
            MAT1 = dict()

            for i, temp in enumerate(MAT):
                for name in MAT.dtype.names:
                    if name == "nr":
                        temp[name] = np.ones(temp["rx"].shape[-1],
                                              dtype=int) * temp["nr"][0][0]

                    if name != "frequencies":
                        if i == 0:
                            MAT1[name] = temp[name]
                        else:
                            MAT1[name] = np.concatenate((MAT1[name],
                                                         temp[name]), axis=-1)
                if self.firstonly:
                    break

            self.f = MAT[0]["frequencies"].ravel()
            sorting = np.argsort(self.f)
            self.rx, self.ry, self.rz = MAT1["rx"]
            self.f = self.f[sorting]
            self.DATA = MAT1["data"][:, sorting, :]
            self.ERR = MAT1["err"][:, sorting, :]
        else:

            self.f = MAT["frequencies"]
            self.rx, self.ry, self.rz = MAT["rx"]
            self.DATA = MAT["data"]
            self.ERR = MAT["err"]

        self.line = MAT["nr"]
        self.alt = self.rz

    def loadWWUMatFile(self, filename):
        """Load data from mat file (WWU processing)."""
        print("Need to implement Anneke's processing")
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

        self.f = np.round(100.0 / np.squeeze(MAT["periods"])) / 100.
        self.ry, self.rx = MAT["xy"]
        if "topo" in MAT.dtype.names:
            self.rz = MAT["topo"][0]
        elif "lla" in MAT.dtype.names:
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
        """Needs MT adaption."""

        # """Simulate data by assuming 1D layered model.

        # Parameters
        # ----------
        # rho : float|iterable
        #     Resistivity (vector)
        # thk : []|float|iterable
        #     Thickness vector (len(thk)+1 must be len(rho))
        # fullTx : bool [False]
        #     Compute full segments (slower)
        # show : bool [False]
        #     Show result
        # **kwargs are passed to the show function
        # """
        # cmp = [1, 1, 1]  # cmp = kwargs.pop("cmp", self.cmp)
        # self.createConfig(fullTx=kwargs.pop("fullTx", False))
        # rho = np.atleast_1d(rho)
        # thk = np.atleast_1d(thk)
        # if len(thk) > 0:
        #     assert len(rho) == len(thk) + 1, "rho/thk lengths do not match"
        #     depth = np.hstack((0., np.cumsum(thk)))
        # else:  # append an artificial layer to enforce RHS
        #     rho = np.hstack((rho, rho[0]))
        #     depth = np.hstack((0., 1000.))

        # DATAX = np.zeros((self.nF, self.nRx), dtype=complex)
        # DATAY = np.zeros_like(DATAX)
        # DATAZ = np.zeros_like(DATAX)
        # for ix in range(self.nRx):
        #     self.setPos(ix)
        #     fop1d = fopSAEM(depth, self.cfg, self.f, cmp)
        #     resp = fop1d.response(rho)

        #     respR, respI = np.reshape(resp, (2, -1))
        #     respC = np.reshape(respR+respI*1j, (3, -1))
        #     DATAX[:, ix] = respC[0, :]
        #     DATAY[:, ix] = respC[1, :]
        #     DATAZ[:, ix] = respC[2, :]

        # self.RESP = np.stack([DATAX, DATAY, DATAZ])
        # if kwargs.pop("show", False):
        #     self.showData(what="response", **kwargs)

    def computePrimaryFields(self):
        """Needs MT adaption."""
        pass
        # """Compute primary fields."""
        # cfg = dict(self.cfg)
        # fak = 4e-7 * np.pi * 1e9  # H->B and T in nT
        # cfg["rec"] = [self.rx, self.ry, self.alt, 0, 0]  # x
        # cfg["freqtime"] = self.f
        # cfg["xdirect"] = True
        # cfg["res"] = [2e14]
        # cfg["depth"] = []
        # print(cfg)
        # pfx = bipole(**cfg).real * fak
        # cfg["rec"][3:5] = [90, 0]  # y
        # pfy = bipole(**cfg).real * fak
        # cfg["rec"][3:5] = [0, 90]  # z
        # pfz = bipole(**cfg).real * fak
        # self.prim = np.stack([pfx, pfy, pfz])

    def txDistance(self, seg=True):  # why should MT have tx distance? !!!
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
        for i in range(3):
            if cmp[i] > 0:
                data = getattr(self, "data"+allcmp[i].upper())
                kwargs.setdefault("color", "C" + str(i))
                kwargs.setdefault("label", "B" + allcmp[i])
                ax = showSounding(data, self.f, ax=ax, ls="",
                                  marker="x", **kwargs)
                if response is not None:
                    col = kwargs["color"]
                    ax[0].plot(respRe[ncmp], self.f, ls="-", color=col)
                    ax[1].plot(respIm[ncmp], self.f, ls="-", color=col)
                    ncmp += 1

        for a in ax:
            a.legend()

        return ax

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
        for iC in range(len(self.cstr)):
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

    def saveData(self, fname=None, line=None, txdir=1, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        if "cmp" in kwargs and kwargs["cmp"] == "all":
            for cmp in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                        [0, 1, 1], [1, 1, 1]]:
                kwargs["cmp"] = cmp
                self.saveData(fname=fname, line=line, txdir=txdir, **kwargs)

        cmp = kwargs.setdefault("cmp", self.cmp)
        allcmp = ['X', 'Y', 'Z']
        if fname is None:
            fname = self.basename
            if line is not None:
                fname += "-line" + str(line)

            for i in range(len(self.cstr)):
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
        data["tx_ids"] = [0]
        DATA = [data]
        meany = 0  # np.median(self.ry[ind]) # needed anymore?
        np.savez(fname+".npz",
                 tx=[np.column_stack((np.array(self.tx)[::txdir],
                                      np.array(self.ty)[::txdir]-meany,
                                      np.array(self.tx)*0))],
                 freqs=self.f,
                 cmp=cmp,
                 DATA=DATA,
                 line=self.line,
                 origin=np.array(self.origin),  # global coordinates w altitude
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

    # def loadResponse(self, dirname=None, response=None):
    #     """Load model response file."""
    #     if response is None:
    #         respfiles = sorted(glob(dirname+"response_iter*.npy"))
    #         if len(respfiles) == 0:
    #             respfiles = sorted(glob(dirname+"reponse_iter*.npy"))  # TYPO
    #         if len(respfiles) == 0:
    #             pg.error("Could not find response file")

    #         responseVec = np.load(respfiles[-1])
    #         respR, respI = np.split(responseVec, 2)
    #         response = respR + respI*1j

    #     sizes = [len(self.cstr), self.nF, self.nRx]
    #     RESP = np.ones(np.prod(sizes), dtype=np.complex) * np.nan

    #     print(self.getIndices(), len(response))
    #     RESP[self.getIndices()] = response
    #     asd
    #     try:
    #         RESP[self.getIndices()] = response
    #     except ValueError:
    #         RESP[:] = response

    #     RESP = np.reshape(RESP, sizes)
    #     self.RESP = np.ones((len(self.cstr), self.nF, self.nRx),
    #                         dtype=np.complex) * np.nan
    #     self.RESP[np.nonzero(self.cmp)[0]] = RESP

    # def getIndices(self):
    #     """Return indices of finite data into full matrix."""
    #     ff = np.array([], dtype=bool)
    #     for i in range(len(self.cstr)):
    #         if self.cmp[i]:
    #             tmp = self.DATA[i].ravel() * self.ERR[i].ravel()
    #             idx = np.argwhere(np.isnan(tmp)).ravel()
    #             mask = np.ones(tmp.size, dtype=bool)
    #             mask[idx] = False
    #             ff = np.hstack((ff, mask[mask==True]))

    #     return ff

    def nData(self):
        """Number of data (for splitting the response)."""
        return sum(self.getIndices())

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


# if __name__ == "__main__":
    # self = MTData(datafile="data_f*.mat")
    # print(self)
    # self.generateDataPDF()
    # self.showData(nf=1)
    # self.showField("alt", background="BKG")
    # self.invertSounding(nrx=20)
    # plotSymbols(self.rx, self.ry, -self.alt, numpoints=0)
    # self.showSounding(nrx=20)
    # self.showData(nf=1)
    # self.generateDataPDF()
