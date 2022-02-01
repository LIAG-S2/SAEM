import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# import pygimli as pg
from pygimli.viewer.mpl import showVecMatrix
from pygimli.core.math import symlog


def deol(s):
    """Remove any EOL from a line."""
    return s.rstrip("\r").rstrip("\n")


def lastint(line):
    """Return integer value from last entry in a line (without EOL)."""
    return int(deol(line).split()[-1])


class Mare2dEMData():
    """Class for holding Mare2dEM data."""

    def __init__(self, filename=None, **kwargs):
        """Initialize class with possible load."""
        self.basename = "new"
        self.f = []
        self.DATA = []
        self.origin = [0, 0, 0]
        self.angle = 0
        self.utmzone = 32
        nameType = {}  # [""] * 50
        xyz = "yxz"  # changed x and y position
        for i in range(3):
            nameType[1+i*2] = "RealE" + xyz[i]
            nameType[2+i*2] = "ImagE" + xyz[i]
            nameType[11+i*2] = "RealB" + xyz[i]
            nameType[12+i*2] = "ImagB" + xyz[i]
            nameType[21+i*2] = "AmpE" + xyz[i]
            nameType[22+i*2] = "PhsE" + xyz[i]
            nameType[27+i] = "log10E" + xyz[i]
            nameType[31+2*i] = "AmpB" + xyz[i]
            nameType[32+2*i] = "PhsB" + xyz[i]
            nameType[37+i] = "log10B" + xyz[i]

        self.nameType = nameType
        self.typeName = {v: k for k, v in self.nameType.items()}
        if filename is not None:
            self.load(filename, **kwargs)

    def __repr__(self):
        """String representation."""
        sdata = "Mare2dEM data with {} data".format(len(self.DATA))
        sf = "{} frequencies ({:.1f}-{:.1f} Hz)".format(
            len(self.f), min(self.f), max(self.f))
        ntx = len(self.txpos) if isinstance(self.txpos,
                                            list) else self.txpos.shape[0]
        stx = "{} transmitters".format(ntx)
        srx = "{} receivers".format(self.rxpos.shape[0])
        sty = "Data types:"
        for ty in np.unique(self.DATA["Type"]):
            sty += " " + self.nameType[ty]

        return "\n".join((sdata, sf, stx + " , " + srx, sty))

    def load(self, filename, flipimag=0, flipxy=False):
        """Load file (.emdata) into class."""
        with open(filename) as fid:
            lines = fid.readlines()
            i = 0
            while "Freq" not in lines[i]:
                if lines[i].startswith("UTM"):
                    sline = lines[i].split()
                    self.origin = [float(sline[-2]), float(sline[-3]), 0]
                    self.angle = float(sline[-1])
                    self.utmzone = int(sline[-5])
                    print("UTM", self.utmzone, self.origin, self.angle)
                elif lines[i].startswith("Phase Convention"):
                    sline = lines[i].split()
                    flipimag = sline[-1].startswith("lag")
                    if flipimag:
                        print("FLipImag lag")

                i += 1

            nf = lastint(lines[i])
            i += 1
            self.f = [float(deol(lines[i+j])) for j in range(nf)]
            i += nf
            nt = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ")
            TX = np.genfromtxt(lines[i:i+nt+1], names=True)
            if flipxy:
                self.txpos = np.column_stack([TX["Y"], TX["X"], -TX["Z"],
                                              TX["Length"], TX["Azimuth"]])
            else:
                self.txpos = np.column_stack([TX["X"], TX["Y"], -TX["Z"],
                                              TX["Length"], TX["Azimuth"]])
            i += nt + 1
            nr = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ")
            RX = np.genfromtxt(lines[i:i+nr+1], names=True)
            if flipxy:
                self.rxpos = np.column_stack([RX["Y"], RX["X"], -RX["Z"]])
            else:
                self.rxpos = np.column_stack([RX["X"], RX["Y"], -RX["Z"]])

            i += nr + 1
            nd = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ").replace("#", " ")
            self.DATA = np.genfromtxt(lines[i:i+nd+1], names=True)
            # self.DATA["Rx"] = self.DATA["Rx"].astype(int)
            # self.DATA["Tx"] = self.DATA["Tx"].astype(int)
            # self.DATA = pd.DataFrame(self.DATA)
            self.basename = filename.replace(".emdata", "")
            # self.basename = filename.rstrip(".emdata")
            # here we should correct all B/logB/E/logE for Tx length!
            if flipimag:
                for i, ty in enumerate(self.DATA["Type"]):
                    if self.nameType[ty].startswith("Phs"):
                        self.DATA["Data"][i] *= -1
                    if self.nameType[ty].startswith("Imag"):
                        self.DATA["Data"][i] *= -1

    def local2global(self, xy):
        """Transform local to global coordinates."""
        ang = np.deg2rad(self.angle)
        A = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]])
        return xy.dot(A) + self.origin[:2]

    def rxPositions(self, globalCoordinates=True):
        """Receiver positions in global coordinates."""
        return self.local2global(
            self.rxpos[:, :2]) if globalCoordinates else self.rxpos[:, :2]

    def txPositions(self, globalCoordinates=False):
        """Return transmitter positions."""
        if isinstance(self.txpos, list):
            return self.txpos
        else:  # if self.txpos.shape[1] > 3:  # x,y,z,len,az,dip
            TX = []
            for it, txi in enumerate(self.txpos):
                x, y, z = txi[:3]
                length = txi[3]
                ang = np.deg2rad(txi[4] if len(txi) > 4 else 0)
                rot = np.array([[np.cos(ang), np.sin(ang)],
                                [-np.sin(ang), np.cos(ang)]])
                pp = rot.dot([[0, 0], [1, -1]]).T * length / 2
                pp[:, 0] += txi[0]
                pp[:, 1] += txi[1]
                pp = np.column_stack((pp, [txi[2], txi[2]]))
                if globalCoordinates:
                    pp = self.local2global(pp[:, :2])

                TX.append(pp)

            return TX

    def showPositions(self, globalCoordinates=False, background="BKG",
                      **kwargs):
        """Show positions."""
        save = kwargs.pop("save", False)
        TX = self.txPositions(globalCoordinates)
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            _, ax = plt.subplots()

        print(ax)
        rxpos = self.rxPositions(globalCoordinates)
        kwargs.setdefault("markersize", 1)
        ax.plot(rxpos[:, 0], rxpos[:, 1], "b.", **kwargs)
        for tx in TX:
            ax.plot(tx[:, 0], tx[:, 1], "r-")

        ax.set_aspect(1.0)
        ax.grid(True)
        if globalCoordinates:
            if background == "BKG":
                from pygimli.viewer.mpl import underlayBKGMap
                underlayBKGMap(ax, utmzone=self.utmzone,
                               uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
            elif background is not None:
                from pygimli.viewer.mpl import underlayMap
                underlayMap(ax, self.utmzone, vendor=background)

        if save:
            ax.figure.savefig(self.basename+"-pos.pdf",
                              bbox_inches="tight", dpi=300)
        return ax

    def nData(self):
        """Number of data."""
        return len(self.DATA)

    def rx(self):
        """Receiver index."""
        return self.DATA["Rx"].astype(int)

    def tx(self):
        """Receiver index."""
        return self.DATA["Tx"].astype(int)

    def filter(self, ePhiMax=None, eAMax=None):
        """Filter according to several criteria.

        Parameters
        ----------
        ePhiMax : float
            maximum phase error (%)
        eAMax : float
            maximum amplitude error (1, e.g. 0.08 means 8%)
        xMin/xMax/yMin/yMax : float
            minimum/maximum x value

        """
        fPhi = [self.nameType[t].startswith("Phs") for t in self.DATA["Type"]]
        fA = np.logical_not(fPhi)
        good = np.isfinite(self.DATA["Data"]*self.DATA["StdErr"])
        errPhi = self.DATA["StdErr"][fPhi]
        good[fPhi] = errPhi < ePhiMax
        errA = self.DATA["StdErr"][fA]
        good[fA] = errA < eAMax
        self.DATA = self.DATA[good]

    def getPart(self, tx=None, typ=None, clean=False):
        """Get a subpart of the data."""
        new = Mare2dEMData()
        new.basename = self.basename + "_"
        for tok in ["f", "txpos", "rxpos", "DATA"]:
            setattr(new, tok, getattr(self, tok))

        if tx is not None:
            bind = new.DATA["Tx"] < 0
            new.basename += "Tx"
            for txi in np.atleast_1d(tx):
                bind = np.bitwise_or(bind, new.DATA["Tx"] == txi)
                new.basename += "{}".format(txi)

            new.DATA = new.DATA[bind]

        if typ is not None:
            if isinstance(typ, str):  # "B", "Bx" or "E"
                typ = [k for k, v in self.nameType.items() if typ in v]
            bind = new.DATA["Type"] < 0
            for typi in np.atleast_1d(typ):
                bind = np.bitwise_or(bind, new.DATA["Type"] == typi)

            new.DATA = new.DATA[bind]

        if clean:
            new.removeUnusedRx()
            new.removeUnusedTx()

        return new

    def removeUnusedRx(self):
        """Remove unused receivers."""
        uR, indF, indB = np.unique(self.rx() - 1,
                                   return_index=True, return_inverse=True)
        self.DATA["Rx"] = indB + 1
        self.rxpos = self.rxpos[uR, :]

    def removeUnusedTx(self):
        """Remove unused receivers."""
        uT, indF, indB = np.unique(self.tx() - 1,
                                   return_index=True, return_inverse=True)
        self.DATA["Tx"] = indB + 1
        if isinstance(self.txpos, list):
            self.txpos = [self.txpos[i] for i in uT]
        else:
            self.txpos = self.txpos[uT, :]

    def getDataMatrix(self, field="Bx", tx=None, column="Data"):
        """Prepare custEM-ready data matrix."""
        mydata = self.getPart(tx=tx, typ=field)

        ut = np.unique(mydata.tx())
        if len(ut) == 0:
            print("No data found!")
            return np.array([])

        assert len(ut) == 1, "More than one transmitter specified!"
        nr = mydata.rx() - 1
        nf = mydata.DATA["Freq"].astype(int) - 1

        amp = np.ones([len(mydata.f), mydata.rxpos.shape[0]]) * np.nan
        phi = np.ones([len(mydata.f), mydata.rxpos.shape[0]]) * np.nan
        re = np.ones([len(mydata.f), mydata.rxpos.shape[0]]) * np.nan
        im = np.ones([len(mydata.f), mydata.rxpos.shape[0]]) * np.nan
        vals = mydata.DATA[column]
        typ = mydata.DATA["Type"]
        atyp = mydata.typeName["log10"+field]
        ptyp = mydata.typeName["Phs"+field]
        rtyp = mydata.typeName["Real"+field]
        ityp = mydata.typeName["Imag"+field]
        for i in range(len(mydata.DATA)):
            if typ[i] == atyp:
                amp[nf[i], nr[i]] = 10**vals[i]
            elif typ[i] == ptyp:
                phi[nf[i], nr[i]] = vals[i]
            elif typ[i] == rtyp:
                re[nf[i], nr[i]] = vals[i]
            elif typ[i] == ityp:
                im[nf[i], nr[i]] = vals[i]

        out = amp * np.exp(np.deg2rad(phi)*1j)
        if not np.any(np.isfinite(out)):
            print("Using real/imag instead")
            out = re + im * 1j
        if np.any(np.isnan(out)):
            print("Found NaN values!")

        return out

    def chooseF(self, find=None, fmin=0, fmax=9e9, every=None):
        """Choose frequencies to be used subsequently."""
        if find is None:
            if every is not None:
                find = np.arange(0, len(self.f), every)
            else:  # fmin/fmax
                find = np.nonzero(np.bitwise_and(np.array(self.f) >= fmin,
                                                 np.array(self.f) <= fmax))[0]

        freq = self.DATA["Freq"].astype(int) - 1
        aind = -np.ones(len(self.f))
        aind[find] = np.arange(len(find))
        self.f = np.take(self.f, find)
        ind = np.in1d(freq, find)
        self.DATA["Freq"] = aind[freq] + 1
        self.DATA = self.DATA[ind]

    def correctTxLengths(self):
        """Correct data by multiplying with the transmitter lengths."""
        pass

    def saveData(self, tx=None, absError=1.5e-3, relError=0.04, topo=1):
        """Save data for inversion with custEM."""
        tx = tx or np.arange(len(self.txpos)) + 1
        DATA = []
        TX = []
        fak = 1e9
        fname = self.basename + "_B_Tx"
        for it, txi in enumerate(np.atleast_1d(tx)):
            # assert matX.shape == matZ.shape, "Bx and Bz not matching"
            if isinstance(self.txpos, list):
                tt = self.txpos[txi-1]
                txl = np.sum(np.sqrt(np.sum(np.diff(tt, axis=0)**2, axis=1)))
                TX.append(tt)
                txl = 1
            else:
                txl = self.txpos[txi-1, 3]
                TX.append(np.column_stack((
                    [self.txpos[txi-1, 0], self.txpos[txi-1, 0]],
                    self.txpos[txi-1, 1] + np.array([-1/2, 1/2])*txl,
                    [self.txpos[txi-1, 2]*topo, self.txpos[txi-1, 2]*topo])))

            part = self.getPart(tx=txi, typ="B", clean=True)
            matX = part.getDataMatrix(field="Bx") * txl * fak
            matY = part.getDataMatrix(field="By") * txl * fak
            matZ = -part.getDataMatrix(field="Bz") * txl * fak
            # errX = part.getDataMatrix(field="Bx", column="Stderr")
            mats = [matX, matY, matZ]
            allcmp = ["x", "y", "z"]
            icmp = [i for i in range(3) if len(mats[i]) > 0]
            if len(icmp) > 0:
                fname += "{}".format(txi)
                dataR = np.zeros([1, len(icmp), *mats[icmp[0]].shape])
                # dataR = np.zeros([1, *mats[icmp[0]].shape, len(icmp)])
                dataI = np.zeros_like(dataR)
                lcmp = 0
                for i in icmp:
                    dataR[0, lcmp, :, :] = mats[i].real
                    dataI[0, lcmp, :, :] = mats[i].imag
                    # dataR[0, :, :, lcmp] = mats[i].real
                    # dataI[0, :, :, lcmp] = mats[i].imag
                    lcmp += 1

                errorR = np.abs(dataR) * relError + absError
                errorI = np.abs(dataI) * relError + absError
                data = dict(dataR=dataR, dataI=dataI,
                            errorR=errorR, errorI=errorI,
                            tx_ids=[int(txi-1)],
                            rx=part.rxpos*np.array([1, 1, topo]),
                            cmp=["B"+allcmp[i] for i in icmp])
                DATA.append(data)

        if len(TX) > 1:
            print(len(TX), TX)
        if len(DATA) > 0:
            np.savez(fname+".npz",
                     tx=TX,
                     freqs=self.f,
                     DATA=DATA,
                     origin=self.origin,  # global coordinates with altitude
                     rotation=self.angle)

    def generateDataPDF(self):
        """Generate a multipage pdf of all data."""
        DATA = self.DATA
        uty = np.unique(DATA["Type"].astype(int))
        ut = np.unique(self.tx())
        fig = plt.figure()
        tol = 1e-5
        # urx = np.unique(self.rxpos[:, 0])
        with PdfPages(self.basename+"-data.pdf") as pdf:
            for tty in uty:
                DA1 = DATA[DATA["Type"] == tty]
                print(self.nameType[tty], sum(DATA["Type"] == tty))
                for it, tt in enumerate(ut):
                    DA2 = DA1[DA1["Tx"] == tt]
                    if DA2.shape[0] > 0:
                        fig.clf()
                        ax = fig.add_subplot(111)
                        dvec = DA2["Data"] * 1.0
                        nam = self.nameType[tty]
                        x = np.round(np.take(self.rxpos[:, 0],
                                             DA2["Rx"].astype(int)-1))
                        y = np.take(self.f, DA2["Freq"].astype(int)-1)
                        if (nam.startswith("RealB") or
                                nam.startswith("ImagB") or
                                nam.startswith("AmpB")):
                            dvec *= 1e9  # nT
                        if nam.startswith("Phs"):
                            dvec = (dvec % 360) - 180
                            kw = dict(cMap="hsv", ax=ax, cMin=-180, cMax=180)
                            if nam[-1] == "y":
                                kw.update(cMin=-45, cMax=45, cMap="jet")
                            elif nam[-1] == "x":
                                dvec[dvec < -45] += 180
                                kw.update(cMin=-45, cMax=45, cMap="jet")

                            showVecMatrix(x, y, dvec, **kw)
                        elif nam.startswith("Real") or nam.startswith("Imag"):
                            showVecMatrix(x, y, symlog(dvec, tol),
                                          cMap="seismic", ax=ax)
                        else:  # nam.startswith("log10"):
                            kw = dict(ax=ax, cMap="Spectral_r",
                                      cMin=-15, cMax=-12)
                            if nam.startswith("log10E"):
                                kw.update(cMin=-10, cMax=-6)

                            showVecMatrix(x, y, dvec, **kw)

                        # ax.set_xlim(0, ur[-1]-1)
                        ax.set_ylim(-0.5, len(self.f)-0.5)
                        ax.xaxis.set_tick_params(rotation=45)
                        ax.set_xlabel("Rx position (m)")
                        ax.set_ylabel("f (Hz)")
                        ax.set_title(self.nameType[tty] + " : " +
                                     "Tx {} (x={:d})".format(
                                         tt, int(self.txpos[it, 0])))
                        ax.figure.savefig(pdf, format='pdf')


if __name__ == "__main__":
    self = Mare2dEMData("P5.emdata")
    print(self)
    self.chooseF(fmax=1000)
    print(self)
    self.chooseF(every=2)
    print(self)
    print(self.f)
    self.basename += "f2"
    self.generateDataPDF()
    self.saveData()
