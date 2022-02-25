"""Plotting tools for SAEM."""
# os and numpy stuff
import numpy as np
# MPL
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.patches import Rectangle
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pygimli.viewer.mpl import underlayMap, underlayBKGMap
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
        ax[0].loglog(np.abs(snddata), freqs, **kwargs)
        ax[1].semilogy(np.angle(snddata)*180/np.pi, freqs, **kwargs)
        ax[0].set_xlabel("|T| (log10 nT/A)")
        ax[1].set_xlabel(r"$\phi$ (°)")
    else:
        ax[0].semilogy(np.real(snddata), freqs, **kwargs)
        ax[1].semilogy(np.imag(snddata), freqs, **kwargs)
        ax[0].set_xlabel("T-real (nT/A)")
        ax[1].set_xlabel("T-imag (nT/A)")

    ax[0].set_ylabel("f (Hz)")

    for a in ax:
        a.grid(True)

    # if response is not None:
        # showSounding(response, "-", ax=ax, amphi=amphi, **kwargs)

    return ax


def plotSymbols(x, y, w, ax=None, **kwargs):
    """Plot circles or rectangles for each point in a map.

    Parameters
    ----------
    x, y : iterable
        x and y positions
    w : iterable
        values to plot
    cmap : mpl.colormap | str ["Spectral"]
        colormap
    colorBar : bool [True]
        draw colowbar
    clim : [float, float]
        min/max values for colorbar
    logScale : bool [False]
        use logarithmic color scaling
    label : str
        label for the colorbar
    radius : float
        prescribing radius of symbol
    numpoints : int
        number of points (0 means circle)
    """
    # kwargs_setdefaults(kwargs)
    cmap = kwargs.pop("cmap", "Spectral_r")
    numpoints = kwargs.pop("numpoints", 0)
    radius = kwargs.pop("radius", 10.)
    label = kwargs.pop("label", False)

    assert len(x) == len(y) == len(w), "Vector lengths have to match!"
    if ax is None:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", ms=0, zorder=-10)

    patches = []
    width = np.min(np.abs(np.diff(x[:len(np.unique(y))])))

    for xi, yi in zip(x, y):
        if numpoints == 0 and type(radius) is not str:
            rect = Circle((xi, yi), radius, ec=None)
        elif radius == 'rect':
            rect = Rectangle((xi-width*0.49, yi), width*0.98, 1.,ec=None)
        else:
            rect = RegularPolygon((xi, yi), numpoints, radius=radius, ec=None)

        patches.append(rect)

    norm = None
    alim = kwargs.pop("alim", [min(w), max(w)])
    log = kwargs.pop("log", False)
    if log:
        norm = SymLogNorm(linthresh=alim[0], vmin=-alim[1], vmax=alim[1])
    else:
        norm = Normalize(vmin=alim[0], vmax=alim[1])

    pc = collections.PatchCollection(patches, cmap=cmap, linewidths=0)
    pc.set_norm(norm)
    pc.set_array(w)
    ax.add_collection(pc)
    if log:
        pc.set_clim([-alim[1], alim[1]])
    else:
        pc.set_clim([alim[0], alim[1]])

    cb = None
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(pc, cax=cax)

    if label:
        cb.set_label(label)

    return ax, cb


def underlayBackground(ax, background="BKG", utm=32):
    """Underlay background from any map."""
    if background == "BKG":
        underlayBKGMap(ax, uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
    else:
        underlayMap(ax, utm, vendor=background)


def makeSymlogTicks(cb, alim):

    i1 = int(np.log10(alim[0]))
    i2 = int(np.log10(alim[1]))
    ni = i2-i1+1
    lvec = np.linspace(i1, i2, ni)
    lvec = np.append([-10**ele for ele in lvec], 0.)
    ticks = np.append(lvec, [10**ele for ele in np.linspace(i2, i1, ni)])
    cb.set_ticks(ticks)
    cb.set_ticklabels(['{:.0e}'.format(tick) for tick in ticks])


def updatePlotKwargs(cmp, **kwargs):
    """Set default values for different plotting tools."""
    cmp = kwargs.setdefault("cmp", cmp)
    what = kwargs.setdefault("what", "data")
    log = kwargs.setdefault("log", True)
    if log:
        cmap = kwargs.setdefault("cmap", "PuOr_r")
        alim = kwargs.setdefault("alim", [1e-3, 1e1])
    else:
        cmap = kwargs.setdefault("cmap", "seismic")
        alim = kwargs.setdefault("alim", [-10., 10.])
    amphi = kwargs.setdefault("amphi", False)

    plim = kwargs.setdefault("plim", [-180., 180.])
    llthres = kwargs.setdefault("llthres", alim[0])

    if log and alim[0] != llthres:
        print("Warning, different values vor *llthres* and *alim[0]* are "
              "usually not reasonbale. Continuing ..." )

    return kwargs