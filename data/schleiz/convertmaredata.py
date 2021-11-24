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

    def __init__(self, filename=None):
        """Initialize class with possible load."""
        self.basename = "new"
        self.f = []
        self.DATA = []
        self.origin = [0, 0, 0]
        self.angle = 0
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
            self.load(filename)

    def __repr__(self):
        """String representation."""
        sdata = "Mare2dEM data with {} data".format(len(self.DATA))
        sf = "{} frequencies ({:.1f}-{:.1f} Hz)".format(
            len(self.f), min(self.f), max(self.f))
        stx = "{} transmitters".format(self.txpos.shape[0])
        srx = "{} receivers".format(self.rxpos.shape[0])
        sty = "Data types:"
        for ty in np.unique(self.DATA["Type"]):
            sty += " " + self.nameType[ty]

        return "\n".join((sdata, sf, stx + " , " + srx, sty))

    def load(self, filename):
        """Load file (.emdata) into class."""
        with open(filename) as fid:
            lines = fid.readlines()
            i = 0
            while "Freq" not in lines[i]:
                i += 1

            nf = lastint(lines[i])
            i += 1
            self.f = [float(deol(lines[i+j])) for j in range(nf)]
            i += nf
            nt = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ")
            TX = np.genfromtxt(lines[i:i+nt+1], names=True)
            self.txpos = np.column_stack([TX["Y"], TX["X"], -TX["Z"],
                                          TX["Length"]])
            i += nt + 1
            nr = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ")
            RX = np.genfromtxt(lines[i:i+nr+1], names=True)
            self.rxpos = np.column_stack([RX["Y"], RX["X"], -RX["Z"]])
            i += nr + 1
            nd = lastint(lines[i])
            i += 1
            lines[i] = lines[i].replace("!", " ").replace("#", " ")
            self.DATA = np.genfromtxt(lines[i:i+nd+1], names=True)
            # self.DATA["Rx"] = self.DATA["Rx"].astype(int)
            # self.DATA["Tx"] = self.DATA["Tx"].astype(int)
            # self.DATA = pd.DataFrame(self.DATA)
            self.basename = filename.rstrip(".emdata")

    def rx(self):
        """Receiver index."""
        return self.DATA["Rx"].astype(int)

    def tx(self):
        """Receiver index."""
        return self.DATA["Tx"].astype(int)

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
        self.txpos = self.txpos[uT, :]

    def getDataMatrix(self, field="Bx", tx=None):
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
        vals = mydata.DATA["Data"]
        typ = mydata.DATA["Type"]
        atyp = mydata.typeName["log10"+field]
        ptyp = mydata.typeName["Phs"+field]
        for i in range(len(mydata.DATA)):
            if typ[i] == atyp:
                amp[nf[i], nr[i]] = 10**vals[i]
            elif typ[i] == ptyp:
                phi[nf[i], nr[i]] = vals[i]

        out = amp * np.exp(np.deg2rad(phi)*1j)
        if np.any(np.isnan(out)):
            print("Found NaN values!")

        return out

    def saveData(self, tx=None, absError=1e-12, relError=0.02, topo=0):
        """Save data for inversion with custEM."""
        if tx is None:
            tx = np.arange(len(self.txpos)) + 1

        DATA = []
        TX = []
        fname = self.basename + "_B_Tx"
        for it, txi in enumerate(np.atleast_1d(tx)):
            part = self.getPart(tx=txi, typ="B", clean=True)
            matX = part.getDataMatrix(field="Bx")
            matZ = part.getDataMatrix(field="Bz")
            assert matX.shape == matZ.shape, "Bx and Bz not matching"
            txl = self.txpos[txi-1, 3]
            TX.append(np.column_stack((
                [self.txpos[txi-1, 0], self.txpos[txi-1, 0]],
                self.txpos[txi-1, 1] + np.array([-1/2, 1/2])*txl,
                [self.txpos[txi-1, 2]*topo, self.txpos[txi-1, 2]*topo])))
            if len(matX) > 0:
                fname += "{}".format(txi)
                dataR = np.zeros([1, *matX.shape, 2])
                dataI = np.zeros([1, *matZ.shape, 2])
                dataR[0, :, :, 0] = matX.real
                dataR[0, :, :, 1] = matZ.real
                dataI[0, :, :, 0] = matX.imag
                dataI[0, :, :, 1] = matZ.imag
                fak = 1e9
                errorR = np.zeros([1, *matX.shape, 2])
                errorR[0, :, :, 0] = np.abs(matX) * relError + absError
                errorR[0, :, :, 1] = np.abs(matZ) * relError + absError
                errorI = errorR
                data = dict(dataR=dataR*fak, dataI=dataI*fak,
                            errorR=errorR*fak, errorI=errorI*fak,
                            tx_ids=[int(txi-1)],
                            rx=part.rxpos*np.array([1, 1, topo]),
                            cmp=["Bx", "Bz"])
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
        # uf = np.unique(DATA["Freq"].astype(int))
        # ur = np.unique(DATA["Rx"].astype(int))
        ut = np.unique(DATA["Tx"].astype(int))
        fig = plt.figure()
        tol = 1e-5
        # urx = np.unique(self.rxpos[:, 0])
        with PdfPages(self.basename+"-data4.pdf") as pdf:
            for tty in uty:
                DA1 = DATA[DATA["Type"] == tty]
                print(self.nameType[tty], sum(DATA["Type"] == tty))
                for it, tt in enumerate(ut[3:4]):
                    DA2 = DA1[DA1["Tx"] == tt]
                    if DA2.shape[0] > 0:
                        fig.clf()
                        ax = fig.add_subplot(111)
                        dvec = DA2["Data"] * 1.0
                        nam = self.nameType[tty]
                        x = np.take(self.rxpos[:, 0], DA2["Rx"].astype(int)-1)
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
                        ax.xaxis.set_tick_params(rotation=90)
                        ax.set_xlabel("Rx number (not position!)")
                        ax.set_ylabel("f (Hz)")
                        ax.set_title(self.nameType[tty] + " : " +
                                     "Tx {} (x={:d})".format(
                                         tt, int(self.txpos[it, 0])))
                        ax.figure.savefig(pdf, format='pdf')

# %%
self = Mare2dEMData("Ball.emdata")
print(self)
if 1:  # frequency decimation to octave-wise (factor 2): 28->10 frequencies
    # %%
    freq = self.DATA["Freq"].astype(int) - 1
    find = np.arange(0, len(self.f), 3)
    aind = -np.ones(len(self.f))
    aind[find] = np.arange(len(find))
    self.f = np.take(self.f, find)
    ind = np.in1d(freq, find)
    self.DATA["Freq"] = aind[freq] + 1
    self.DATA = self.DATA[ind]
    print(self)

# %% save files for every single transmitter and whole
for i in range(6):
    self.saveData(tx=i+1)

self.saveData()
# %%
if 0:
    # %%
    TX1B = self.getPart(tx=1, typ="B", clean=True)  # , clean=True)
    print(TX1B)
    TX1B.saveData()
    # %%
    mat = TX1B.getDataMatrix("Bx")
    plt.matshow(np.abs(mat))
    plt.matshow(np.angle(mat))
    # %%
    TX2B = self.getPart(tx=[1, 2], typ="B", clean=True)  # , clean=True)
    print(TX2B)
    # %%
    TX3 = self.getPart(tx=3, clean=True)
    mat = TX3.getDataMatrix("Bx")
    # %%
    TX1E = self.getPart(tx=1, typ="E", clean=True)  # , clean=True)  # , typ=)
    print(TX1E)
    # %%
    TX1 = self.getPart(tx=1)  # , clean=True)  # , typ=)
    print(TX1)
    uR, indF, indB = np.unique(TX1.DATA["Rx"].astype(int) - 1,
                               return_index=True, return_inverse=True)
    TX1.DATA["Rx"] = indB + 1
    TX1.rxpos = TX1.rxpos[uR, :]
    print(TX1)
    # %%
    # %% check for duplicate sensors and their ordering
    _, indF, indB = np.unique(rxpos[:, 0], return_index=True, return_inverse=True)
    rxSorted = rxpos[indF, :]
    i = 111
    print(rxpos[i, :], rxSorted[indB[i], :])
