"""EMData base class for any type of electromagnetic data."""
from glob import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyproj

import pygimli as pg
from .plotting import plotSymbols, underlayBackground, makeSymlogTicks
from .plotting import makeSubTitles, updatePlotKwargs
from .tools import detectLinesAlongAxis, detectLinesBySpacing
from .tools import detectLinesByDistance, detectLinesOld
from .tools import readCoordsFromKML, is_point_inside_polygon, distToTx


class EMData():
    """Class for EM frequency-domain data."""

    def __init__(self, **kwargs):
        """Initialize CSEM data class.

        Parameters
        ----------
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
        self.origin = [0., 0., 0.]
        self.angle = 0.
        self.llthres = kwargs.pop("llthres", 1e-3)
        self.A = np.array([[1, 0], [0, 1]])
        self.depth = None
        self.DATA = None  # better np.array([]) ?
        self.PRIM = None
        self.RESP = None
        self.ERR = None
        self.mode = None  # needs to be done in derived classes
        self.f = kwargs.pop("f", [])
        self.basename = kwargs.pop("basename", "noname")
        zone = kwargs.pop("zone", 32)
        self.verbose = kwargs.pop("verbose", True)
        self.utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')

        self.rx = kwargs.pop("rx", np.array([0.]))
        self.ry = kwargs.pop("ry", np.zeros_like(self.rx))
        if isinstance(self.ry, (int, float)):
            self.ry = np.ones_like(self.rx)*self.ry
        if isinstance(self.rx, (int, float)):
            self.rx = np.ones_like(self.ry)*self.rx
        self.rz = kwargs.pop("rz", np.ones_like(self.rx)*kwargs.pop("alt", 0.))
        if isinstance(self.rz, (int, float)):
            self.rz = np.ones_like(self.rx)*self.rz
        self.line = kwargs.pop("line", np.ones_like(self.rx, dtype=int))
        self.cmp = []
        self.cstr = []
        self.radius = 1
        self.txAlt = 0 # rather ground altitude

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

    def txDistance(self):
        """Dummy tx Distance (to be overwritten in CSEM)."""
        return np.zeros(self.nRx)

    def getIndices(self):
        """Return indices of finite data into full matrix."""
        ff = np.array([], dtype=bool)
        for i in range(len(self.cstr)):
            if self.cmp[i]:
                tmp = self.DATA[i].ravel() * self.ERR[i].ravel()
                ff = np.hstack((ff, np.isfinite(tmp)))

        return ff

    def nData(self):
        """Number of data (for splitting the response)."""
        return sum(self.getIndices())

    def chooseActive(self, what="data"):
        """Choose activate data for visualization.

        If what is an array of correct shape instead of a str, individual data
        arrays can be passed to the visualization methods.


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
                sf - measured secondary field
                sf/pf - measured secondary field divided by primary fields
        """
        if isinstance(what, str):
            what = what.lower()
            self.cb_label='[nT/A]'
            if (what.find("pf") >= 0 or what.find("sf") >= 0
                and self.PRIM is None):
                self.computePrimaryFields()
            if what == "data":
                return self.DATA[:]
            elif what.startswith("resp"):
                return self.RESP[:]
            elif what == "aerr" or what == "abserror":
                return self.ERR[:]
            elif what == "rerr" or what == "relerror":
                rr = self.ERR.real / (np.abs(self.DATA.real) + 1e-12)
                ii = self.ERR.imag / (np.abs(self.DATA.imag) + 1e-12)
                return rr + ii * 1j
            elif what == "amisfit":
                return self.DATA[:] - self.RESP[:]
            elif what == "rmisfit":
                tmp = np.zeros_like(self.DATA)
                for i in range(len(self.DATA)):
                    rr = (1. - self.RESP[i].real / self.DATA[i].real) * 100.
                    ii = (1. - self.RESP[i].imag / self.DATA[i].imag) * 100.
                    tmp[i, :] = rr + ii * 1j
                    self.cb_label='[%]'
                return tmp
            elif what == "wmisfit":
                mis = self.DATA - self.RESP
                wmis = mis.real / self.ERR.real + mis.imag / self.ERR.imag * 1j
                self.cb_label=''
                return wmis
            elif what == "pf":
                return self.PRIM[:]
            elif what == "sf":
                return self.DATA - self.PRIM
            elif what == "sf/pf":
                self.cb_label = ''
                return (self.DATA - self.PRIM) / np.abs(self.PRIM)
            else:
                print('Error! Wrong argument chosen to specify active data. '
                      'Aborting  ...')
                raise SystemExit
        else:
            print('  -  choosing provided argument as data array  -  ')
            return what

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

        self.cfg["rec"][:3] = self.rx[nrx], self.ry[nrx], self.rz[nrx]-self.txAlt
        self.dataX = self.DATA[0, :, nrx]
        self.dataY = self.DATA[1, :, nrx]
        self.dataZ = self.DATA[2, :, nrx]
        self.nrx = nrx
        if show:
            self.showPositions()

    def createConfig(self, fullTx=False):
        """Create EMPYMOD input argument configuration."""
        self.cfg = {'rec': [self.rx[0], self.ry[0], self.rz[0], 0, 90],
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
            Bxy = self.A.T.dot(np.vstack((self.DATA[0, i, :],
                                          self.DATA[1, i, :])))
            self.DATA[0, i, :] = Bxy[0, :]
            self.DATA[1, i, :] = Bxy[1, :]
            "Need to correct Error rotation!"
            # Errxy = self.A.T.dot(np.vstack((self.ERR[0, i, :],
            #                                 self.ERR[1, i, :])))
            # self.ERR[0, i, :] = Bxy[0, :]
            # self.ERR[1, i, :] = Bxy[1, :]

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

        if origin is not None:
            self.setOrigin()
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

            # print('Need to fix field rotation of X/Y components')
            for i in range(len(self.f)):
                Bxy = self.A.dot(np.vstack((self.DATA[0, i, :],
                                            self.DATA[1, i, :])))
                self.DATA[0, i, :] = Bxy[0, :]
                self.DATA[1, i, :] = Bxy[1, :]
                "Need to correct Error rotation!"
                # Errxy = self.A.dot(np.vstack((self.ERR[0, i, :],
                #                               self.ERR[1, i, :])))
                # self.ERR[0, i, :] = Errxy[0, :]
                # self.ERR[1, i, :] = Errxy[1, :]

        self.createConfig()  # make sure rotated Tx is in cfg
        self.angle = ang
        if origin is not None:
            self.setOrigin(shift_back=True)

    def setOrigin(self, origin=None, shift_back=True):
        """Set origin."""
        # first shift back to old origin
        # origin = origin or self.origin
        # if shift_back:
        #     self.tx += self.origin[0]
        #     self.ty += self.origin[1]
        #     self.rx += self.origin[0]
        #     self.ry += self.origin[1]
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

    def filter(self, f=-1, fmin=0, fmax=1e6, fInd=None, nInd=None, rInd=None,
               minTxDist=None, maxTxDist=None, every=None, line=None,
               polygon=None):
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
        rInd : iterable, optional
            index array for receivers to remove
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
        polygon : ndarray|str
            polygone (or kmlfile) to remove points
            * inside (minTxDist not set) OR
            * in a distance of minTxDist
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
            self.PRIM = self.PRIM[:, fInd, :]

        # part 2: receiver axis
        if nInd is None:
            if polygon is not None:
                if isinstance(polygon, str):
                    polygon = readCoordsFromKML(polygon).T

                rx = self.rx + self.origin[0]
                ry = self.ry + self.origin[1]
                if minTxDist is None:  # inside
                    nInd = np.ones_like(rx, dtype=bool)
                    for i, xy in enumerate(zip(rx, ry)):
                        nInd[i] = not is_point_inside_polygon(*xy, polygon)
                else:
                    di = distToTx(rx, ry, polygon[:, 0], polygon[:, 1])
                    nInd = np.nonzero((di >= minTxDist))[0]
            elif minTxDist is not None or maxTxDist is not None:
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

        if rInd is not None:
            nInd = np.delete(np.arange(len(self.rx)), rInd)

        if nInd is not None:
            for tok in ['alt', 'rx', 'ry', 'rz', 'line']:
                setattr(self, tok, getattr(self, tok)[nInd])

            self.DATA = self.DATA[:, :, nInd]
            if np.any(self.ERR):
                self.ERR = self.ERR[:, :, nInd]
            if np.any(self.RESP):
                self.RESP = self.RESP[:, :, nInd]
            if np.any(self.PRIM):
                self.PRIM = self.PRIM[:, :, nInd]
            if hasattr(self, 'MODELS'):
                self.MODELS = self.MODELS[nInd, :]

    def skinDepths(self, rho=30):
        """Compute skin depth based on a medium resistivity."""
        return np.sqrt(rho/self.f) * 500

    def createDepthVector(self, rho=30, nl=15):
        """Create depth vector."""
        sd = self.skinDepths(rho=rho)
        self.depth = np.hstack([0, pg.utils.grange(min(sd)*0.3, max(sd)*1.2,
                                                   n=nl, log=True)])

    def showPositions(self, ax=None, line=None, background=None, org=False,
                color=None, marker=None, **kwargs):
        """Show receiver positions."""
        if ax is None:
            _, ax = plt.subplots()

        rxy = np.column_stack((self.rx, self.ry))
        txy = np.column_stack((self.tx, self.ty))
        if org:
            rxy += self.origin[:2]
            txy += self.origin[:2]

        kwargs.setdefault("markersize", 5)
        ma = marker or "."
        ax.plot(rxy[:, 0], rxy[:, 1], ma, markersize=2,
                color=color or "blue")
        if txy.shape[1] > 0:
            if np.any(txy):
                ax.plot(txy[:, 0], txy[:, 1], "-", markersize=4,
                        color=color or "orange")

        if hasattr(self, "nrx") and self.nrx < self.nRx:
            ax.plot(rxy[self.nrx, 0], rxy[self.nrx, 1], "ro", **kwargs)

        if line is not None:
            ax.plot(rxy[self.line == line, 0],
                    rxy[self.line == line, 1], "-")

        ax.set_aspect(1.0)
        ax.grid(True)
        ax.set_xlabel("Easting (m) UTM32N")
        ax.set_ylabel("Northing (m) UTM32N")
        if background:
            underlayBackground(ax, background, self.utm)

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
            _, ax = plt.subplots()

        kwargs.setdefault("radius", self.radius)
        kwargs.setdefault("log", False)
        kwargs.setdefault("cmap", "Spectral_r")
        background = kwargs.pop("background", None)
        ax.plot(self.rx, self.ry, "k.", ms=1, zorder=-10)
        if np.any(self.tx) or np.any(self.ty):
            ax.plot(self.tx, self.ty, "k*-", zorder=-1)

        if isinstance(field, str):
            kwargs.setdefault("label", field)
            if field == "txDist":
                field = self.txDistance()
            else:
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

        if "poly" in kwargs and poly is not None:
            poly = kwargs["poly"]
            if isinstance(poly, str):  # only a single
                poly = [readCoordsFromKML(poly)]
            elif isinstance(poly, np.ndarray):
                poly = [poly]
            elif isinstance(poly, list):  #
                if isinstance(poly[0], str):
                    poly = [readCoordsFromKML(p).T for p in poly]

            for p in poly:
                ax.plot(p[:, 0]-self.origin[0], p[:, 1]-self.origin[1])

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
        kw = updatePlotKwargs(**kwargs)
        label = kwargs.setdefault("label", kw["what"])
        lw = kwargs.setdefault("lw", 0.5)
        cmp = kwargs.setdefault("cmp", self.cmp)
        axis = kwargs.setdefault("axis", "x")
        nn, x = self.sortAlongAxis(line, axis)
        DATA = self.chooseActive(kw["what"])

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2,
                                   squeeze=False, sharex=True,
                                   sharey=not kw["amphi"],
                                   figsize=kwargs.pop("figsize", (12, 8)))
        else:
            fig = ax.flat[0].figure

        errbar = None
        if kw["what"].lower() == 'data' and np.any(self.ERR):
            errbar = self.ERR[:, nf, nn]

        ncmp = 0
        for ci, cid in enumerate(cmp):
            if cid:
                subset = DATA[ci, nf, nn]
                if kw["amphi"]:  # amplitude and phase
                    ax[0, ncmp].plot(x, np.abs(subset), label=label)
                    ax[1, ncmp].plot(x, np.angle(subset, deg=True),
                                     label=label)
                    if kw["log"]:
                        ax[0, ncmp].set_yscale('log')
                        ax[0, ncmp].set_ylim(kw["alim"])
                        ax[1, ncmp].set_ylim(kw["plim"])
                    ax[0, ncmp].set_title(r'|| (' + self.cstr[ci] + ') ||')
                    ax[1, ncmp].set_title(r'$\phi$(' + self.cstr[ci] + ')')
                    print('Warning! No error bars and response comparison '
                          'implemented yet for amplitude and phase plots with '
                          'the linefreq method. Continuing  ...')
                else:  # real and imaginary part
                    if kw["what"] == 'data' and errbar is not None:
                        ax[0, ncmp].errorbar(
                            x, np.real(subset),
                            yerr=[errbar[ci].real, errbar[ci].real],
                            marker='o', lw=0., barsabove=True,
                            color=kw["color"],
                            elinewidth=0.5, markersize=3, label=label)
                        ax[1, ncmp].errorbar(
                            x, np.imag(subset),
                            yerr=[errbar[ci].imag, errbar[ci].imag],
                            marker='o', lw=0., barsabove=True,
                            color=kw["color"],
                            elinewidth=0.5, markersize=3, label=label)
                    else:
                        ax[0, ncmp].plot(x, np.real(subset), 'x-', lw=lw,
                                         color=kw["color"], label=label)
                        ax[1, ncmp].plot(x, np.imag(subset), 'x-', lw=lw,
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
                    ax[0, ncmp].set_title(r'$\Re$(' + self.cstr[ci] + ')')
                    ax[1, ncmp].set_title(r'$\Im$(' + self.cstr[ci] + ')')
                ncmp += 1

        if kw["amphi"]:
            ax[0, 0].set_ylabel("Amplitude (nT/A)")
            ax[1, 0].set_ylabel("Phase (°)")
        else:
            if kw["field"] == 'B':
                ax[0, 0].set_ylabel("(nT/A)")
                ax[1, 0].set_ylabel("(nT/A)")
            elif kw["field"] == 'E':
                ax[0, 0].set_ylabel("(V/m)")
                ax[1, 0].set_ylabel("(V/m)")

        for a in ax[-1, :]:
            if axis == "x":
                a.set_xlim([np.min(self.rx), np.max(self.rx)])
            elif axis == "y":
                a.set_xlim([np.min(self.ry), np.max(self.ry)])
            elif axis == "d":
                print('Warning! Need to adjust xlim for *x* = *d* option')
            a.set_xlabel(axis + ' (m)')

        for a in ax.flat:
            a.set_aspect('auto')
            a.grid(True)

        ax.flat[0].legend()

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)
        return fig, ax

    def showLineData(self, line=None, ax=None, **kwargs):
        """Show alternative line plot."""
        kw = updatePlotKwargs(**kwargs)
        kw.setdefault("radius", "rect")
        cmp = kwargs.setdefault("cmp", self.cmp)
        axis = kwargs.setdefault("axis", "x")
        nn, _ = self.sortAlongAxis(line, axis)
        DATA = self.chooseActive(kw["what"])

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2,
                                   squeeze=False, sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (12, 8)))
        else:
            fig = ax.flat[0].figure

        if axis == "x":
            x = np.tile(self.rx[nn], len(self.f))
        elif axis == "y":
            x = np.tile(self.ry[nn], len(self.f))
        elif axis == "d":
            print('need to implement *d* option')
            # need to eval line direction first, otherwise bugged
            # x = np.sqrt((self.rx[nn]-self.rx[0])**2+
            #             (self.ry[nn]-self.ry[0])**2)
            # x = np.sqrt((np.mean(self.tx)-self.rx[nn])**2+
            #             (np.mean(self.ty)-self.ry[nn])**2)
        y = np.repeat(np.arange(len(self.f)), len(nn))

        ncmp = 0
        for ci, cid in enumerate(cmp):
            if cid:
                subset = DATA[ci, :, nn].T.ravel()
                if kw["amphi"]:
                    kw["cmap"] = 'viridis'
                    plotSymbols(x, y, np.abs(subset), ax=ax[0, ncmp],
                                mode="amp", **kw)
                    plotSymbols(x, y, np.angle(subset, deg=1), ax=ax[1, ncmp],
                                mode="phase", **kw)
                    ax[0, ncmp].set_title(r'|| (' + self.cstr[ci] + ') ||')
                    ax[1, ncmp].set_title(r'$\phi$(' + self.cstr[ci] + ')')
                else:
                    _, cb1 = plotSymbols(x, y, np.real(subset),
                                         ax=ax[0, ncmp], **kw)
                    _, cb2 = plotSymbols(x, y, np.imag(subset),
                                         ax=ax[1, ncmp], **kw)

                    for cb in [cb1, cb2]:
                        if ncmp + 1 == sum(cmp) and kw["log"]:
                            makeSymlogTicks(cb, kw["alim"])
                        elif ncmp + 1 == sum(cmp) and not kw["log"]:
                            pass
                        else:
                            cb.set_ticks([])
                    ax[0, ncmp].set_title(r'$\Re$(' + self.cstr[ci] + ')')
                    ax[1, ncmp].set_title(r'$\Im$(' + self.cstr[ci] + ')')
                ncmp += 1

        ax[0, 0].set_ylim([0., len(self.f)])
        for a in ax[-1, :]:
            if axis == "x":
                a.set_xlim([np.min(self.rx), np.max(self.rx)])
            elif axis == "y":
                a.set_xlim([np.min(self.ry), np.max(self.ry)])
            elif axis == "d":
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

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)
        return fig, ax

    def showLineDataMat(self, line=None, ax=None, **kwargs):
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
        kw = updatePlotKwargs(**kwargs)
        cmp = kwargs.setdefault("cmp", self.cmp)
        axis = kwargs.setdefault("axis", "x")
        nn, _ = self.sortAlongAxis(line, axis)
        DATA = self.chooseActive(kw["what"])

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2,
                                   squeeze=False, sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (12, 8)))
        else:
            fig = ax.flat[0].figure

        ncmp = 0
        for ci, cid in enumerate(cmp):
            if cid:
                subset = DATA[ci, :, nn].T
                if kw["amphi"]:  # amplitude and phase
                    norm = LogNorm(vmin=kw["alim"][0], vmax=kw["alim"][1])
                    pc1 = ax[0, ncmp].matshow(np.abs(subset), norm=norm,
                                              cmap="viridis")
                    if kw["alim"] is not None:
                        pc1.set_clim(kw["alim"])
                    pc2 = ax[1, ncmp].matshow(np.angle(subset, deg=True),
                                              cmap="hsv")
                    pc2.set_clim(kw["plim"])
                    ax[0, ncmp].set_title(r'|| (' + self.cstr[ci] + ') ||')
                    ax[1, ncmp].set_title(r'$\phi$(' + self.cstr[ci] + ')')
                else:  # real and imaginary part
                    if kw["log"]:
                        snorm = SymLogNorm(vmin=-kw["alim"][1],
                                           vmax=kw["alim"][1],
                                           linthresh=kw["alim"][0])
                        pc1 = ax[0, ncmp].matshow(
                            np.real(subset), norm=snorm, cmap=kw["cmap"])
                        pc2 = ax[1, ncmp].matshow(
                            np.imag(subset), norm=snorm, cmap=kw["cmap"])
                    else:
                        pc1 = ax[0, ncmp].matshow(np.real(subset),
                                                  cmap=kw["cmap"])
                        if kw["alim"] is not None:
                            pc1.set_clim([kw["alim"][0], kw["alim"][1]])
                        pc2 = ax[1, ncmp].matshow(np.imag(subset),
                                                  cmap=kw["cmap"])
                        if kw["alim"] is not None:
                            pc2.set_clim([kw["alim"][0], kw["alim"][1]])

                        ax[0, ncmp].set_title(r'$\Re$(' + self.cstr[ci] + ')')
                        ax[1, ncmp].set_title(r'$\Im$(' + self.cstr[ci] + ')')

                for j, pc in enumerate([pc1, pc2]):
                    divider = make_axes_locatable(ax[j, ncmp])
                    cax = divider.append_axes("right", size="5%", pad=0.15)
                    cb = plt.colorbar(pc, cax=cax, orientation="vertical")
                    if not kw["amphi"]:
                        if ncmp + 1 == sum(cmp) and kw["log"]:
                            makeSymlogTicks(cb, kw["alim"])
                        elif ncmp + 1 == sum(cmp) and not kw["log"]:
                            pass
                        else:
                            cb.set_ticks([])
                ncmp += 1

        ax[0, 0].set_ylim([-0.5, len(self.f)-0.5])
        yt = np.arange(0, len(self.f), 2)
        ytl = ["{:.0f}".format(self.f[yy]) for yy in yt]
        for aa in ax[:, 0]:
            aa.set_yticks(yt)
            aa.set_yticklabels(ytl)
            aa.set_ylabel("f (Hz)")

        if axis == "x":
            xt = np.round(np.linspace(0, len(nn)-1, 7))
            xtl = ["{:.0f}".format(self.rx[nn[int(xx)]]) for xx in xt]
        elif axis == "y":
            xt = np.round(np.linspace(0, len(nn)-1, 7))
            xtl = ["{:.0f}".format(self.ry[nn[int(xx)]]) for xx in xt]
        elif axis == "d":
            print('Warning! Need to adjust xlim for *x* = *d* option')

        for aa in ax[-1, :]:
            if xtl[0] > xtl[-1]:
                aa.set_xlim([xt[1], xt[0]])
            aa.set_xticks(xt)
            aa.set_xticklabels(xtl)
            aa.set_xlabel(kwargs['axis'] + ' (m)')

        for a in ax.flat:
            a.set_aspect('auto')

        name = kwargs.pop("name", self.basename)
        if "what" in kwargs:
            name += " " + kwargs["what"]

        fig.suptitle(name)
        return fig, ax

    def showPatchData(self, nf=0, ax=None, background=None, **kwargs):
        """Show all three components as amp/phi or real/imag plots.

        Parameters
        ----------
        nf : int | float
            frequency index (int) or value (float) to plot
        """
        kw = updatePlotKwargs(**kwargs)
        kw.setdefault("numpoints", 0)
        kw.setdefault("radius", self.radius)
        cmp = kwargs.setdefault("cmp", self.cmp)
        DATA = self.chooseActive(kw["what"])

        if isinstance(nf, float):
            nf = np.argmin(np.abs(self.f - nf))

        if ax is None:
            fig, ax = plt.subplots(ncols=sum(cmp), nrows=2, squeeze=False,
                                   sharex=True, sharey=True,
                                   figsize=kwargs.pop("figsize", (12, 8)))
        else:
            fig = ax.flat[0].figure

        poly = kwargs.pop("poly", None)
        if isinstance(poly, str):  # only a single
            poly = [readCoordsFromKML(poly)]
        elif isinstance(poly, np.ndarray):
            poly = [poly]
        elif isinstance(poly, list):  #
            if isinstance(poly[0], str):
                poly = [readCoordsFromKML(p).T for p in poly]

        ncmp = 0
        for ci, cid in enumerate(cmp):
            if cid:
                subset = DATA[ci, nf, :]
                if kw["amphi"]:
                    kw["cmap"] = 'Spectral_r'
                    _, cb1 = plotSymbols(self.rx, self.ry, np.abs(subset),
                                ax=ax[0, ncmp], mode="amp", **kw)
                    _, cb2 = plotSymbols(self.rx, self.ry, np.angle(subset, deg=1),
                                ax=ax[1, ncmp], mode="phase", **kw)
                    ax[0, ncmp].set_title(r'|| (' + self.cstr[ci] + ') ||')
                    ax[1, ncmp].set_title(r'$\phi$(' + self.cstr[ci] + ')')
                    cb1.ax.set_title(self.cb_label)
                    cb2.ax.set_title('[°]')
                else:
                    _, cb1 = plotSymbols(self.rx, self.ry, np.real(subset),
                                         ax=ax[0, ncmp], **kw)
                    _, cb2 = plotSymbols(self.rx, self.ry, np.imag(subset),
                                         ax=ax[1, ncmp], **kw)

                    makeSubTitles(ax, ncmp, self.cstr, ci, kw["what"])
                    for cb in [cb1, cb2]:
                        if ncmp + 1 == sum(cmp) and kw["log"]:
                            if kw["symlog"]:
                                makeSymlogTicks(cb, kw["alim"])
                        elif ncmp + 1 == sum(cmp) and not kw["log"]:
                            pass
                        else:
                            cb.set_ticks([])
                        cb.ax.set_title(self.cb_label)
                ncmp += 1

        if background is not None and kwargs.pop("overlay", False):  # bwc
            background = "BKG"

        for a in ax.flat:
            a.set_aspect(1.0)
            if np.any(self.tx) or np.any(self.ty):
                a.plot(self.tx, self.ty, "k*-")

            a.plot(self.rx, self.ry, ".", ms=0, zorder=-10)
            if poly is not None:
                for p in poly:
                    a.plot(p[:, 0]-self.origin[0], p[:, 1]-self.origin[1])

            if background:
                underlayBackground(a, background, self.utm)

        for i in range(2):
            ax[i, 0].set_ylabel('[m]')
        for i in range(ncmp):
            ax[1, i].set_xlabel('[m]')

        basename = kwargs.pop("name", self.basename)
        if "what" in kwargs and isinstance(kwargs["what"], str):
            basename += " " + kwargs["what"]

        fig.suptitle(basename+"  f="+str(self.f[nf])+"Hz")
        return fig, ax

    def showData(self, mat=True, *args, **kwargs):
        """Generic show function.

        Upon keyword arguments given, directs to
        * showDataPatch [default]: x-y patch plot for every f
        * showLineData (if line=given): x-f patch plot
        * showLineFreq (if line and nf given): x-f line plot
        """
        if isinstance(mat, str):  # obviously what meant
            kwargs.setdefault('what', mat)
            mat = True
        if "line" in kwargs:
            if "nf" in kwargs:
                return self.showLineFreq(*args, **kwargs)
            else:
                if mat:
                    return self.showLineDataMat(*args, **kwargs)
                else:
                    return self.showLineData(*args, **kwargs)
        else:
            return self.showPatchData(*args, **kwargs)

    def showDataFit(self, line=1, nf=0):
        """Show data and model response for single line/freq."""
        fig, ax = self.showLineFreq(line=line, nf=nf)
        self.showLineFreq(line=line, nf=nf, ax=ax, what="response")
        return fig, ax

    def sortAlongAxis(self, line, axis):
        """Dort points along a given axis (x, y or d)."""
        nn = np.arange(len(self.rx))
        if line is not None:
            nn = np.nonzero(self.line == line)[0]

        if axis == "x":
            x = np.sort(self.rx[nn])
            si = np.argsort(self.rx[nn])
        elif axis == "y":
            x = np.sort(self.ry[nn])
            si = np.argsort(self.ry[nn])
        elif axis == "d":
            # need to eval line direction first, otherwise bugged
            x = np.sort(np.sqrt((np.mean(self.tx) - self.rx[nn])**2 +
                                (np.mean(self.ty) - self.ry[nn])**2))
        else:
            print('Error, "axis" must be "x", "y", or "d". Aborting  ...')
            raise SystemExit
        return(nn[si], x)

    def getData(self, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.setdefault("cmp", self.cmp)
        if np.shape(self.ERR) != np.shape(self.DATA):
            self.estimateError(**kwargs)

        if line is None:  # take all existing (nonzero) lines
            nn = np.nonzero(self.line > 0)[0]
        else:
            nn = np.nonzero(self.line == line)[0]

        ypos = np.round((self.ry[nn])*10)/10  # get to straight line
        rxpos = np.round(np.column_stack((self.rx[nn], ypos,
                                          self.rz[nn]-self.txAlt))*10)/10

        dataR = np.zeros((1, sum(cmp), self.nF, len(nn)))
        dataI = np.zeros_like(dataR)
        errorR = np.zeros_like(dataR)
        errorI = np.zeros_like(dataR)

        cstr = []
        ncmp = 0
        for ic, cid in enumerate(cmp):
            if cid:
                dataR[0, ncmp, :, :] = self.DATA[ic][:, nn].real
                dataI[0, ncmp, :, :] = self.DATA[ic][:, nn].imag
                errorR[0, ncmp, :, :] = self.ERR[ic][:, nn].real
                errorI[0, ncmp, :, :] = self.ERR[ic][:, nn].imag
                cstr.append(self.cstr[ic])
                ncmp += 1

        # error estimation
        data = dict(dataR=dataR, dataI=dataI,
                    errorR=errorR, errorI=errorI,
                    rx=rxpos, cmp=cstr)

        return data

    def saveData(self, fname=None, line=None, txdir=1, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        if "cmp" in kwargs and kwargs["cmp"] == "all":
            for cmp in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                        [0, 1, 1], [1, 1, 1]]:
                kwargs["cmp"] = cmp
                self.saveData(fname=fname, line=line, txdir=txdir, **kwargs)

        cmp = kwargs.setdefault("cmp", self.cmp)
        if fname is None:
            fname = self.basename
            if line is not None:
                fname += "_line" + str(line)

            for ci, cid in enumerate(cmp):
                if cid:
                    fname += self.cstr[ci]
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
        np.savez(fname+".npz",
                 tx=[np.column_stack((np.array(self.tx)[::txdir],
                                      np.array(self.ty)[::txdir],
                                      np.array(self.tz)[::txdir]))],
                 freqs=self.f,
                 DATA=DATA,
                 line=self.line,
                 origin=np.array(self.origin),  # global coordinates w altitude
                 rotation=self.angle)

    def generateDataPDF(self, pdffile=None, figsize=[12, 6],
                        mode='patchwise', **kwargs):
        """Generate a multi-page pdf file containing all data."""
        what = kwargs.setdefault('what', 'data')
        sw = what.replace("/", "_")
        if mode == 'patchwise':
            pdffile = pdffile or self.basename + "_" + sw + ".pdf"
        elif mode == 'linewise':
            pdffile = pdffile or self.basename + "_line_" + sw + ".pdf"
        elif mode == 'linewisemat':
            pdffile = pdffile or self.basename + "_linemat_" + sw + ".pdf"
        elif mode == 'linefreqwise':
            pdffile = pdffile or self.basename + "_linefreqs_" + sw + ".pdf"
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
                            kwargs["what"] = "resp"
                            fig, ax = self.showLineFreq(li, fi, ax=ax,
                                                        **kwargs)
                            fig.suptitle('line = {:.0f}, '
                                         'freq = {:.0f} Hz'.format(li, freq))
                            fig.savefig(pdf, format='pdf')
                            plt.close(fig)

            elif mode == 'linewise' or mode == 'linewisemat':
                fig, ax = plt.subplots(figsize=figsize)
                self.showField(self.line, ax=ax)
                ax.figure.savefig(pdf, format="pdf")

                ul = np.unique(self.line)
                plt.close(fig)
                for li in ul[ul > 0]:
                    nn = np.nonzero(self.line == li)[0]
                    if np.isfinite(li) and len(nn) > 3:
                        if mode == 'linewise':
                            fig, ax = self.showLineData(li, **kwargs)
                        else:
                            fig, ax = self.showLineDataMat(li, **kwargs)
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
        if np.shape(self.DATA) != np.shape(self.ERR):
            self.ERR = np.zeros_like(self.DATA)
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
            self.ERR[cmp, freq, :].real = np.maximum(np.maximum(
                self.ERR[cmp, freq, :].real, aErr[cmp, freq, :].real),
                rErr[cmp, freq, :].real)
            # self.ERR[cmp, freq, :].imag = np.maximum(np.maximum(
            #     self.ERR[cmp, freq, :].imag, aErr[cmp, freq, :].imag),
            #     rErr[cmp, freq, :].imag)
        else:
            self.ERR[cmp, freq, :] = self.ERR[cmp, freq, :] +\
                aErr[cmp, freq, :] + rErr[cmp, freq, :]

    def mask(self, aErr=None, rErr=None, aMin=None, aMax=None,
             pMin=None, pMax=None):
        """Masking out data according to several properties.

        Parameters
        ----------
        aErr : float
            maximum absolute error
        rErr : float
            maximum relative error
        aMin, aMax : float
            minimum/maximum amplitude
        pMin, pMax : float
            minimum/maximum phase
        """
        cnan = np.nan + 1j * np.nan
        if aErr is not None:
            self.DATA[np.abs(self.DATA.real) < aErr] = cnan
            self.DATA[np.abs(self.DATA.imag) < aErr] = cnan

        if rErr is not None:
            rr = np.abs(self.ERR.real) / (np.abs(self.DATA.real) + 1e-12)
            self.DATA[rr > rErr] = cnan
            ii = np.abs(self.ERR.imag) / (np.abs(self.DATA.imag) + 1e-12)
            self.DATA[ii > rErr] = cnan

        if aMin is not None or aMax is not None:
            aa = np.abs(self.DATA)
            if aMin is not None:
                self.DATA[aa < aMin] = cnan
            if aMax is not None:
                self.DATA[aa > aMax] = cnan

        if pMin is not None or pMax is not None:
            pp = np.angle(self.DATA)
            if pMin is not None:
                self.DATA[pp < pMin] = cnan
            if pMax is not None:
                self.DATA[pp > pMax] = cnan


    # for compatibility, forward to mask with good defaults
    def deactivateNoisyData(self, aErr=1e-4, rErr=0.5):
        """Set data below a certain threshold to nan (inactive)."""
        self.mask(aErr=aErr, rErr=rErr)

    def loadResponse(self, dirname=None, response=None):
        """Load model response file."""
        if response is None or type(response) is int:
            if type(response) is int:
                respfiles = [dirname + "response_iter_" +
                             str(response) + ".npy"]
            else:
                respfiles = sorted(glob(dirname+"response_iter*.npy"))
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
        self.RESP = np.ones((len(self.cstr), self.nF, self.nRx),
                            dtype=np.complex) * np.nan
        self.RESP[np.nonzero(self.cmp)[0]] = RESP

    def showSpatialMisfit(self, what="wmisfit", **kwargs):
        """Show spatial distribution of misfit plots.

        chi-square over components, frequency and Real/Imag.

        Parameters
        ----------
        what : str ['wmisfit]
            property to integrate over
        log : bool [False]
            use logscale colorbar
        kwargs : dict
            keyword args passed to showField/plotSymbols

        Returns
        -------
        ax, cb : matplotlib Axes and colorbar objects
        """
        mis = self.chooseActive(what=what)
        mR = np.nanmean(mis.real**2, axis=(0, 1))
        mI = np.nanmean(mis.imag**2, axis=(0, 1))
        kwargs.setdefault('symlog', False)
        return self.showField((mR+mI)/2, **kwargs)

    def showMisfitStats(self, what="wmisfit", **kwargs):
        """Show misfit statistics for data components.

        Chi-square integrated over space, as a function
        of components, frequency and real/imag parts.

        Parameters
        ----------
        what : str ['wmisfit]
            property to integrate over
        log : bool [False]
            use logscale colorbar
        vmin, vmax : float [min/max values]
            minimum and maximum colorbar ranges
        log : bool [False]
            use logarithmic colorbar
        cmap : str ['Spetral_r']
            matplotlib colormap or string

        Returns
        -------
        ax : [Axes, Axes]
            two matplotlib axes for real and imaginary
        """
        mis = self.chooseActive(what=what)
        statR = np.nanmean(mis.real**2, axis=2)
        statI = np.nanmean(mis.imag**2, axis=2)
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
        kwargs.setdefault("vmin", min(np.min(statR), np.min(statI)))
        kwargs.setdefault("vmax", min(np.max(statR), np.max(statI)))
        kwargs.setdefault("cmap", "Spectral_r")
        if kwargs.pop("log", True):
            kwargs["vmin"] = np.log10(kwargs["vmin"])
            kwargs["vmax"] = np.log10(kwargs["vmax"])
            statR = np.log10(statR)
            statI = np.log10(statI)
        for i, ri in enumerate([statR, statI]):
            im = ax[i].imshow(ri, **kwargs)
            plt.colorbar(im, ax=ax[i])
            ax[i].set_yticks([0, 1, 2], ["Bx", "By", "Bz"])

        ax[1].set_xticks(range(self.nF),
                         [str(int(fi)) for fi in self.f])
        ax[0].set_title("Real")
        ax[1].set_title("Imag")
        ax[1].set_xlabel("f (Hz)")
        return ax
