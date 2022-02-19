import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from saem import Mare2dEMData
from saem import CSEMData


class CSEMSurvey():
    """Class for (multi-patch/transmitter) CSEM data."""

    def __init__(self, arg=None, **kwargs):
        """Initialize class with either data filename or patch instances.


        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.


        """
        self.patches = []
        self.origin = [0, 0]
        self.angle = 0
        self.basename = "new"
        self.cmp = [1, 1, 1]
        if arg is not None:
            if isinstance(arg, str):
                if arg.endswith(".emdata"):  # obviously a Mare2d File
                    self.importMareData(arg, **kwargs)

    def __repr__(self):
        st = "CSEMSurvey class with {:d} patches".format(len(self.patches))
        for i, p in enumerate(self.patches):
            st = "\n".join([st, p.__repr__()])

        return st

    def importMareData(self, mare, txs=None):
        """Import Mare2dEM file format."""
        if isinstance(mare, str):
            return self.importMareData(Mare2dEMData(mare), txs=txs)

        ntx = len(mare.txPositions())
        for i in range(ntx):
            part = mare.getPart(tx=i+1, typ="B", clean=True)
            txl = mare.txpos[i, 3]
            txpos = np.column_stack((
                [mare.txpos[i, 1], mare.txpos[i, 1]],
                mare.txpos[i, 0] + np.array([-1/2, 1/2])*txl))
            fak = 1e9
            matX = part.getDataMatrix(field="Bx") * txl * fak
            matY = part.getDataMatrix(field="By") * txl * fak
            matZ = -part.getDataMatrix(field="Bz") * txl * fak
            rx, ry, rz = part.rxpos.T
            cs = CSEMData(f=mare.f, rx=rx, ry=ry, rz=rz,
                          txPos=txpos)
            cs.DATA = np.stack((matX, matY, matZ))
            cs.chooseData()
            self.addPatch(cs)
        if txs:
            for i, p in enumerate(self.patches):
                p.tx, p.ty = txs[i].T[:2]

    def addPatch(self, patch):
        """Add a new patch to the file.

        Parameters
        ----------
        patch : CSEMData | str
            CSEMData instance or string to load into that
        """
        if isinstance(patch, str):
            patch = CSEMData(patch)

        self.patches.append(patch)

    def showPositions(self):
        """Show all positions."""
        fig, ax = plt.subplots()
        ma = ["x", "+"]
        for i, p in enumerate(self.patches):
            p.showPos(ax=ax, color="C{:d}".format(i), marker=ma[i % 2])

        return fig, ax

    def showData(self, **kwargs):
        """."""
        for i, p in enumerate(self.patches):
            p.showData(**kwargs)

    def getData(self, line=None, **kwargs):
        """Gather data from individual patches."""
        DATA = []
        lines = np.array([])
        for i, p in enumerate(self.patches):
            data = p.getData(line=line, **kwargs)
            data["tx_ids"] = [i]  # for now fixed, later global list
            DATA.append(data)
            lines = np.concatenate((lines, p.line+(i+1)*100))

        return DATA, lines

    def saveData(self, fname=None, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.pop("cmp", self.patches[0].cmp)
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

        txs = [np.column_stack((p.tx, p.ty, p.tx*0)) for p in self.patches]
        DATA, lines = self.getData(line=line, **kwargs)
        np.savez(fname+".npz",
                 tx=txs,
                 freqs=self.patches[0].f,
                 DATA=DATA,
                 line=lines,
                 origin=self.origin,  # global coordinates with altitude
                 rotation=self.angle)


if __name__ == "__main__":
    # %% way 1 - load ready patches
    self = CSEMSurvey()
    self.addPatch("Tx1.npz")
    self.addPatch("Tx2.npz")
    print(self)
    self.showPositions()
    p = self.patches[0]
    # %% way 2 - patch instances
    # not so interesting
    # %% way 3 - directly from Mare file
    self.saveData()
