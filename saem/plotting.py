"""Plotting tools for SAEM."""
# os and numpy stuff
import numpy as np
# MPL
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.patches import Circle, RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns


def showSounding(snddata, freqs, ma="rx", ax=None, amphi=True, response=None,
                 **kwargs):
    """Show amplitude and phase data."""
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True)

    if snddata.dtype == np.float:
        snddata = snddata[:len(snddata)//2] + snddata[len(snddata)//2:] * 1j
        # print(len(freqs), len(data))

    if amphi:
        ax[0].loglog(np.abs(snddata), freqs, ma, **kwargs)
        ax[1].semilogy(np.angle(snddata)*180/np.pi, freqs, ma, **kwargs)
        ax[0].set_xlabel("|T| (log10 nT/A)")
        ax[1].set_xlabel(r"$\phi$ (°)")
    else:
        ax[0].semilogy(np.real(snddata), freqs, ma, **kwargs)
        ax[1].semilogy(np.imag(snddata), freqs, ma, **kwargs)
        ax[0].set_xlabel("T-real (nT/A)")
        ax[1].set_xlabel("T-imag (nT/A)")

    ax[0].set_ylabel("f (Hz)")

    for a in ax:
        a.grid(True)

    # if response is not None:
        # showSounding(response, "-", ax=ax, amphi=amphi, **kwargs)

    return ax


def plotSymbols(x, y, w, ax=None, cMap="Spectral",
                clim=None, radius=10, numpoints=0, colorBar=True):
    """Plot circles or rectangles for each point in a map.

    Parameters
    ----------
    x, y : iterable
        x and y positions
    w : iterable
        values to plot
    cmap : mpl.colormap | str
        colormap
    clim : (float, float)
        min/max values for colorbar
    radius : float
        prescribing radius of symbol
    numpoint : int
        number of points (0 means circle)
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", ms=0, zorder=-10)

    if clim is None:
        clim = [min(w), max(w)]

    patches = []
    for xi, yi in zip(x, y):
        if numpoints == 0:
            rect = Circle((xi, yi), radius, ec=None)
        else:
            rect = RegularPolygon((xi, yi), numpoints, radius=radius, ec=None)

        patches.append(rect)

    pc = collections.PatchCollection(patches, cmap=cMap, linewidths=0)
    pc.set_array(w)
    ax.add_collection(pc)
    pc.set_clim(clim)
    if colorBar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pc, cax=cax)

    return pc
