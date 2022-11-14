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
from .tools import readCoordsFromKML, distToTx, detectLinesAlongAxis
from .tools import detectLinesBySpacing, detectLinesByDistance, detectLinesOld


class EMData():
    """Class for EM frequency-domain data."""

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

        self.updateData(**kwargs)
        self.origin = [0, 0, 0]
        self.angle = 0
        self.llthres = 1e-3
        self.A = np.array([[1, 0], [0, 1]])
        self.PRIM = None
        self.RESP = None
        self.ERR = None

    def __repr__(self):
        """String representation of the class."""
        sdata = "EM data with {:d} stations and {:d} frequencies".format(
            len(self.rx), len(self.f))

        return "\n".join((sdata))

    @property
    def nRx(self):
        """Number of receiver positions."""
        return len(self.rx)

    @property
    def nF(self):
        """Number of frequencies."""
        return len(self.f)

    def updateData(self, **kwargs):

        self.f = kwargs.pop("f", [])
        self.basename = "noname"
        self.basename = kwargs.pop("basename", self.basename)
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        self.cmp = kwargs.pop("cmp", [1, 1, 1])  # active components

        self.rx = kwargs.pop("rx", np.array([0.]))
        self.ry = kwargs.pop("ry", np.array([0.]))
        self.rz = kwargs.pop("rz", np.array([0.]))
        dxy = np.sqrt(np.diff(self.rx)**2 + np.diff(self.ry)**2)
        self.radius = np.median(dxy) * 0.5
        self.line = kwargs.pop("line", np.ones_like(self.rx, dtype=int))

        self.tx, self.ty = np.array([0., 0.]), np.array([0., 0.])
        if "txPos" in kwargs:
            txpos = kwargs["txPos"]
            if isinstance(txpos, str):
                if txpos.lower().find(".kml") > 0:
                    self.tx, self.ty, *_ = readCoordsFromKML(txpos)
                else:
                    self.tx, self.ty = np.genfromtxt(txpos, unpack=True,
                                                     usecols=[0, 1])
            else:  # take it directly
                self.tx, self.ty, *_ = np.array(txpos)

    def chooseActive(self, what="data"):
        """
        Choose activate data for visualization. If what is an array of
        correct shape instead of a str, individual data arrays can be passed
        to the visualization methods.


        Parameters
        ----------
        what : str
            property name or matrix to choose / show
                data - measured data
                resp - forward response
                aerr - absolute data error
                rerr - relative data error
                amisfit - absolute misfit between data and response
                rmisfit - relative misfit between data and response
                wmisfit - error-weighted misfit
                pf - primary fields
                sf - measured secondary data divided by primary fields
        """

        self.ACTIVE = np.zeros_like(self.DATA)
        if isinstance(what, str):
            if what.lower() == "data":
                self.ACTIVE[:] = self.DATA[:]
            elif what.lower() == "resp":
                self.ACTIVE[:] = self.RESP[:]
            elif what.lower() == "aerr":
                self.ACTIVE[:] = self.ERR[:]
            elif what.lower() == "rerr":
                rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-12)
                ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-12)
                self.ACTIVE[:] = rr + ii * 1j
            elif what.lower() == "amisfit":
                self.ACTIVE[:] = self.DATA[:] - self.RESP[:]
            elif what.lower() == "rmisfit":
                for i in range(len(self.DATA)):
                    rr = (1. - self.RESP[i].real / self.DATA[i].real) * 100.
                    ii = (1. - self.RESP[i].imag / self.DATA[i].imag) * 100.
                    self.ACTIVE[i, :] = rr + ii * 1j
            elif what.lower() == "wmisfit":
                mis = self.DATA - self.RESP
                wmis = mis.real / self.ERR.real + mis.imag / self.ERR.imag * 1j
                self.ACTIVE[:] = wmis
            elif what.lower() == "pf":
                self.ACTIVE[:] = self.PRIM[:]
            elif what.lower() == "sf":
                primAbs = np.sqrt(np.sum(self.PRIM**2, axis=0))
                self.ACTIVE[:] = self.DATA / primAbs - 1.0
        else:
            print('  -  choosing passed data array  -  ')
            self.ACTIVE[:] = what

    def setPos(self, nrx=0, position=None, show=False):
        """Set the position of the current sounding to be shown or inverted."""
        if hasattr(nrx, '__iter__'):  # obviously a position
            position = nrx
        if position:
            dr = (self.rx - position[0])**2 + (self.ry - position[1])**2
            nrx = np.argmin(dr)
            if self.verbose:
                print("closest point at distance is ", min(np.sqrt(dr)))
                print("Tx distance ", self.txDistance()[nrx])

        self.cfg["rec"][:3] = self.rx[nrx], self.ry[nrx], self.alt[nrx]
        self.dataX = self.DATAX[:, nrx]
        self.dataY = self.DATAY[:, nrx]
        self.dataZ = self.DATAZ[:, nrx]
        self.nrx = nrx
        if show:
            self.showPos()

    def createConfig(self, fullTx=False):
        """Create EMPYMOD input argument configuration."""
        self.cfg = {'rec': [self.rx[0], self.ry[0], self.alt[0], 0, 90],
                    'strength': 1, 'mrec': True,
                    'srcpts': 5,
                    'htarg': {'pts_per_dec': 0, 'dlf': 'key_51_2012'},
                    'verb': 1}
        if fullTx:  # sum up over segments
            self.cfg['src'] = [self.tx[:-1], self.tx[1:],
                               self.ty[:-1], self.ty[1:],
                               -0.1, -0.1]
        else:  # only first&last point (quick)
            self.cfg['src'] = [self.tx[0], self.tx[-1],
                               self.ty[0], self.ty[-1],
                               -0.1, -0.1]

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
        self.setOrigin(origin)
        if ang is None:
            if line is not None:  # use Tx orientation so that Tx points to y
                rx = self.rx[self.line == line]
                ry = self.ry[self.line == line]
                ang = np.median(np.arctan2(ry-ry[0], rx-rx[0]))
            else:  # use specific line so that line points to x
                ang = np.arctan2(self.ty[-1]-self.ty[0],
                                 self.tx[-1]-self.tx[0]) + np.pi / 2
                # ang = np.mean(np.arctan2(
                #     np.diff(self.ty), np.diff(self.tx))) + np.pi / 2

        co, si = np.cos(ang), np.sin(ang)
        self.A = np.array([[co, si], [-si, co]])
        if ang != 0:
            self.tx, self.ty = self.A.dot(np.array([self.tx, self.ty]))
            self.tx = np.round(self.tx*10+0.001) / 10  # just in 2D case
            self.rx, self.ry = self.A.dot(np.vstack([self.rx, self.ry]))

            for i in range(len(self.f)):
                Bxy = self.A.T.dot(np.vstack((self.DATAX[i, :],
                                              self.DATAY[i, :])))
                self.DATAX[i, :] = Bxy[0, :]
                self.DATAY[i, :] = Bxy[1, :]

        self.createConfig()  # make sure rotated Tx is in cfg
        self.angle = ang

    def setOrigin(self, origin=None):
        """Set origin."""
        # first shift back to old origin
        self.tx += self.origin[0]
        self.ty += self.origin[1]
        self.rx += self.origin[0]
        self.ry += self.origin[1]
        if origin is None:
            origin = [np.mean(self.tx), np.mean(self.ty)]
        # now shift to new origin
        self.tx -= origin[0]
        self.ty -= origin[1]
        self.rx -= origin[0]
        self.ry -= origin[1]
        self.origin = origin

    def detectLines(self, mode=None, axis='x', show=False):
        """Split data in lines for line-wise processing.

        Several modes are available:
            'x'/'y': along coordinate axis
            spacing vector: by given spacing
            float: minimum distance
        """

        if isinstance(mode, (str)):
            self.line = detectLinesAlongAxis(self.rx, self.ry, axis=mode)
        elif hasattr(mode, "__iter__"):
            self.line = detectLinesBySpacing(self.rx, self.ry, mode, axis=axis)
        elif isinstance(mode, (int, float)):
            self.line = detectLinesByDistance(self.rx, self.ry, mode,
                                              axis=axis)
        else:
            self.line = detectLinesOld(self.rx, self.ry)

        self.line += 1

        if show:
            self.showField(self.line)

    def removeNoneLineData(self):
        """Remove data not belonging to a specific line."""
        self.filter(nInd=np.nonzero(self.line)[0])

    def filter(self, f=-1, fmin=0, fmax=1e6, fInd=None, nInd=None,
               minTxDist=None, maxTxDist=None, every=None, line=None):
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
        line : int
            remove a line completely
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

        if np.any(self.PRIM):
            for i in range(3):
                self.PRIM[i] = self.PRIM[i][fInd, :]

        # part 2: receiver axis
        if nInd is None:
            if minTxDist is not None or maxTxDist is not None:
                dTx = self.txDistance()
                minTxDist = minTxDist or 0
                maxTxDist = maxTxDist or 9e9
                nInd = np.nonzero((dTx >= minTxDist) * (dTx <= maxTxDist))[0]
            elif line is not None:
                nInd = np.nonzero(self.line != line)[0]
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
            if np.any(self.PRIM):
                for i in range(3):
                    self.PRIM[i] = self.PRIM[i][:, nInd]
            if hasattr(self, 'MODELS'):
                self.MODELS = self.MODELS[nInd, :]
            if self.PRIM is not None:
                for i in range(3):
                    self.PRIM[i] = self.PRIM[i][:, nInd]

        self.chooseData("data")  # make sure DATAX/Y/Z have correct size

    def mask(self, **kwargs):
        """Masking out data according to several properties."""
        pass  # not yet implemented

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
        kw = updatePlotKwargs(kwargs.pop("cmp", self.cmp), **kwargs)
        label = kwargs.pop("label", kw["what"])
        lw = kwargs.pop("lw", 0.5)
        self.chooseData(kw["what"], kw["llthres"])
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

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
            x = np.sort(self.rx[nn])
            si = np.argsort(self.rx[nn])
        elif kwargs["x"] == "y":
            x = np.sort(self.ry[nn])
            si = np.argsort(self.ry[nn])
        elif kwargs["x"] == "d":
            # need to eval line direction first, otherwise bugged
            x = np.sort(np.sqrt((np.mean(self.tx) - self.rx[nn])**2 +
                                (np.mean(self.ty) - self.ry[nn])**2))

        nn = nn[si]
        errbar = None
        if kw["what"] == 'data' and np.any(self.ERR):
            errbar = self.ERR[:, nf, nn]

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
                            color=kw["color"],
                            elinewidth=0.5, markersize=3, label=label)
                        ax[1, ncmp].errorbar(
                            x, np.imag(data),
                            yerr=[errbar[i].imag, errbar[i].imag],
                            marker='o', lw=0., barsabove=True,
                            color=kw["color"],
                            elinewidth=0.5, markersize=3, label=label)
                    else:
                        ax[0, ncmp].plot(x, np.real(data), '--', lw=lw,
                                         color=kw["color"], label=label)
                        ax[1, ncmp].plot(x, np.imag(data), '--', lw=lw,
                                         color=kw["color"], label=label)
                    if kw["log"]:
                        ax[0, ncmp].set_yscale('symlog',
                                               linthresh=kw["llthres"])
                        ax[0, ncmp].set_ylim([-kw["alim"][1], kw["alim"][1]])
                        ax[1, ncmp].set_yscale('symlog',
                                               linthresh=kw["llthres"])
                        ax[1, ncmp].set_ylim([-kw["alim"][1], kw["alim"][1]])
                    else:
                        pass

                ax[0, ncmp].set_title("B"+allcmp[i])
                ncmp += 1
        if kw["amphi"]:
            ax[0, 0].set_ylabel("Amplitude (nT/A)")
            ax[1, 0].set_ylabel("Phase (°)")
        else:
            if kw["field"] == 'B':
                ax[0, 0].set_ylabel(r"$\Re$(B) (nT/A)")
                ax[1, 0].set_ylabel(r"$\Im$(B) (nT/A)")
            elif kw["field"] == 'E':
                ax[0, 0].set_ylabel(r"$\Re$(E) (V/m)")
                ax[1, 0].set_ylabel(r"$\Im$(E) (V/m)")

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

    def showDataFit(self, line=1, nf=0):
        """Show data and model response for single line/freq."""
        fig, ax = self.showLineFreq(line=line, nf=nf)
        self.showLineFreq(line=line, nf=nf, ax=ax, what="response")

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
        kw = updatePlotKwargs(kwargs.pop("cmp", self.cmp), **kwargs)
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
        kw = updatePlotKwargs(kwargs.pop("cmp", self.cmp), **kwargs)
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
        alim = kw.pop("alim", [1e-3, 1])
        plim = kw.pop("plim", [-180, 180])
        for j, cc in enumerate(allcmp):
            data = getattr(self, "DATA"+cc.upper()).copy()
            if scale:
                data /= self.PRIM[j]
            if kw["amphi"]:
                kw.pop("cmap", None)
                kw.pop("log", None)
                plotSymbols(self.rx, self.ry, np.abs(data[nf]),
                            ax=ax[0, j],  colorBar=(j == len(allcmp)-1),
                            **kw,
                            cmap=amap, log=True, alim=alim)
                plotSymbols(self.rx, self.ry, np.angle(data[nf], deg=1),
                            ax=ax[1, j],  colorBar=(j == len(allcmp)-1),
                            **kw,
                            cmap="hsv", log=False, alim=plim)
                ax[0, j].set_title("log10 T"+cc+" [pT/A]")
                ax[1, j].set_title(r"$\phi$"+cc+" [°]")
            else:
                _, cb1 = plotSymbols(self.rx, self.ry, np.real(data[nf]),
                                     ax=ax[0, j], alim=alim, **kw)
                _, cb2 = plotSymbols(self.rx, self.ry, np.imag(data[nf]),
                                     ax=ax[1, j], alim=alim, **kw)

                ax[0, j].set_title("real T"+cc+" [nT/A]")
                ax[1, j].set_title("imag T"+cc+" [nT/A]")

                for cb in [cb1, cb2]:
                    if ncmp + 1 == sum(kw["cmp"]) and kw["log"]:
                        makeSymlogTicks(cb, alim)
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
        if "what" in kwargs and isinstance(kwargs["what"], str):
            basename += " " + kwargs["what"]

        fig.suptitle(basename+"  f="+str(self.f[nf])+"Hz")

        # self.chooseData(kw.get("what", "data"), kw["llthres"])
        if "what" in kwargs:
            self.chooseData("data", kw["llthres"])

        return fig, ax

    def showLineData2(self, line=None, ax=None, **kwargs):
        """Show alternative line plot."""
        kw = updatePlotKwargs(kwargs.pop("cmp", self.cmp), **kwargs)
        self.chooseData(kw["what"], kw["llthres"])
        kw.setdefault("radius", "rect")
        kwx = kw.pop('x', "x")
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
            if kwx == "x":
                a.set_xlim([np.min(self.rx), np.max(self.rx)])
            elif kwx == "y":
                a.set_xlim([np.min(self.ry), np.max(self.ry)])
            elif kwx == "d":
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

        # if "what" in kwargs:
            # self.chooseData("data", kw["llthres"])

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)

        return fig, ax

    def showData(self, *args, **kwargs):
        """Generic show function.

        Upon keyword arguments given, directs to
        * showDataPatch [default]: x-y patch plot for every f
        * showLineData (if line=given): x-f patch plot
        * showLineFreq (if line and nf given): x-f line plot
        """
        if "line" in kwargs:
            if "nf" in kwargs:
                return self.showLineFreq(*args, **kwargs)
            else:
                if kwargs.get("amphi", False):
                    return self.showLineData(*args, **kwargs)
                else:
                    return self.showLineData2(*args, **kwargs)
        else:
            return self.showPatchData(*args, **kwargs)

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

    def estimateError(self, ignoreErr=True, useMax=False, ri=None,
                      **kwargs):
        """Estimate data error to be saved in self.ERR.

        Errors can be (according to useMax=False/True for A/B)
        A) a sum of absolute and relative error (and the processing error)
        B) the maximum of all contributions (relative, absolute, processing)

        Parameters
        ----------
        relError : float [0.05]
            relative data error (0.05 means 5%)
        absError : [self.llthres = 0.001]
            absolute data error in nT/A
        ignoreErr : bool [True]
            ignore already existing error (from processing or previous estimat)
        useMax : bool [False]
            use maximum of all three error parts instead of sum
        cmp : iterable|int [0:3]
            components (0=x, 1=y, 2=z) to which it is applied
        freq : iterable|int [0:nF]
            frequency number(s) to which it is applied
        """
        absError = kwargs.pop("absError", self.llthres)
        relError = kwargs.pop("relError", 0.05)
        cmp = kwargs.pop("cmp", slice(0, 3))
        freq = kwargs.pop("freq", slice(0, self.nF))
        if self.ERR is None:  # never initialized (e.g. after simulate)
            self.ERR = np.zeros_like(self.DATA, dtype=complex)

        if ri is None:
            aErr = np.zeros_like(self.DATA, dtype=complex)
            aErr.real = absError
            aErr.imag = absError
            rErr = np.abs(self.DATA.real) * relError + \
                np.abs(self.DATA.imag) * relError * 1j

            if ignoreErr:
                self.ERR[cmp, freq, :] = 0 + 0j
                #np.zeros_like(self.DATA[cmp, freq, :]) + (0+0j)

        elif ri == "real":
            aErr = np.zeros_like(self.DATA, dtype=complex)
            aErr.real = absError
            rErr = np.abs(self.DATA.real) * relError + \
                np.abs(self.DATA.imag) * 0. * 1j

            if ignoreErr:
                self.ERR[cmp, freq, :].real = np.zeros(
                    self.ERR[cmp, freq, :].real.shape)

        elif ri == "imag":
            aErr = np.zeros_like(self.DATA, dtype=complex)
            aErr.imag = absError
            rErr = np.abs(self.DATA.real) * 0. + \
                np.abs(self.DATA.imag) * relError * 1j

            if ignoreErr:
                self.ERR[cmp, freq, :].imag = np.zeros(
                    self.ERR[cmp, freq, :].real.shape)

        # decide upon adding or maximizing errors
        if useMax:
            self.ERR[cmp, freq, :] = np.maximum(np.maximum(
                self.ERR[cmp, freq, :], aErr[cmp, freq, :]),
                rErr[cmp, freq, :])
        else:
            self.ERR[cmp, freq, :] = self.ERR[cmp, freq, :] +\
                aErr[cmp, freq, :] + rErr[cmp, freq, :]

    def deactivateNoisyData(self, aErr=1e-4, rErr=0.5):
        """Set data below a certain threshold to nan (inactive)."""
        if aErr is not None:
            self.DATA[np.abs(self.DATA.real) < aErr] = np.nan + 1j * np.nan
            self.DATA[np.abs(self.DATA.imag) < aErr] = np.nan + 1j * np.nan

        if rErr is not None:
            rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-12)
            ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-12)

            self.DATA[np.abs(rr) > rErr] = np.nan + 1j * np.nan
            self.DATA[np.abs(ii) > rErr] = np.nan + 1j * np.nan


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
