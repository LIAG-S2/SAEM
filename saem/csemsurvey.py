import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from .maredata import Mare2dEMData
from .saem import CSEMData


class CSEMSurvey():
    """Class for (multi-patch/transmitter) CSEM data."""

    def __init__(self, **kwargs):
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

    def __repr__(self):
        st = "CSEMSurvey class with {:d} patches".format(len(self.patches))
        for i, p in enumerate(self.patches):
            st = "\n".join([st, p.__repr__()])

        return st

    def importMareData(self, marefile):
        """Import Mare2dEM file format."""
        mare = Mare2dEMData(marefile)

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
        for i, p in enumerate(self.patches):
            p.showPos(ax=ax, color="C{:d}".format(i))


if __name__ == "__main__":
    # %% way 1
    self = CSEMSurvey()
    self.addPatch("Tx1.npz")
    self.addPatch("Tx2.npz")
    print(self)
    self.showPositions()
    p = self.patches[0]


