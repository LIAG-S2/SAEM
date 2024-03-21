"""Plotting tools for SAEM."""
# os and numpy stuff
import numpy as np
# MPL
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.patches import Rectangle
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pygimli.viewer.mpl import underlayMap, underlayBKGMap
# import seaborn as sns


def dMap(cmap="Spectral"):
    """Double (mirrored) spectral colormap."""
    # colors = np.vstack((cm, Spectral(np.linspace(0., 1, 128)),
    #                     cm.Spectral_r(np.linspace(0., 1, 128))))
    cm1 = getattr(cm, cmap)
    if cmap.endswith("_r"):
        cmap = cmap[:-2]
    else:
        cmap = cmap + "_r"

    cm2 = getattr(cm, cmap)
    ls = np.linspace(0., 1, 128)
    colors = np.vstack((cm1(ls), cm2(ls)))
    return LinearSegmentedColormap.from_list("mycolors", colors)


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


def plotSymbols(x, y, w, ax=None, mode=None, **kwargs):
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
    log : bool [False]
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
    symlog = kwargs.setdefault("symlog", True)

    assert len(x) == len(y) == len(w), "Vector lengths have to match!"
    if ax is None:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", ms=0, zorder=-10)

    patches = []
    uy = np.unique(y)
    maxn = len(uy) if len(uy) > 1 else len(x)
    width = np.min(np.abs(np.diff(x[:maxn])))

    for xi, yi in zip(x, y):
        if numpoints == 0 and type(radius) is not str:
            rect = Circle((xi, yi), radius, ec=None)
        elif radius == 'rect':
            rect = Rectangle((xi-width*0.5, yi), width, 1., ec=None)
        else:
            rect = RegularPolygon((xi, yi), numpoints, radius=radius, ec=None)

        patches.append(rect)

    norm = None
    if mode == "phase":
        alim = kwargs.setdefault("plim", [-180., 180.])
        cmap = "hsv"
        log = False
    else:
        alim = kwargs.setdefault("alim", [min(w), max(w)])
        log = kwargs.setdefault("log", False)

    if log:
        norm = SymLogNorm(linthresh=alim[0], vmin=-alim[1], vmax=alim[1])
        if not symlog:
            norm = LogNorm(vmin=alim[0], vmax=alim[1])
    else:
        norm = Normalize(vmin=alim[0], vmax=alim[1])

    pc = collections.PatchCollection(patches, cmap=cmap, linewidths=0)
    pc.set_norm(norm)
    if symlog:
        pc.set_array(w)
    else:
        pc.set_array(np.abs(w))
        ax.plot(x[w < 0], y[w < 0], 'k_', markersize=1.)

    ax.add_collection(pc)
    if log:
        pc.set_clim([-alim[1], alim[1]])
        if mode == "amp" or not symlog:
            pc.set_clim(alim[0], alim[1])
    else:
        pc.set_clim([alim[0], alim[1]])

    cb = None
    if kwargs.get("colorBar", True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(pc, cax=cax)

        if label:
            cb.set_label(label)

    return ax, cb


def underlayBackground(ax, background="BKG", utm=32):
    """Underlay background from any map."""
    if background in ["DOP", "DTK", "MAP"]:
        underlayBKGMap(ax, mode=background,
                       uuid='8102b4d5-7fdb-a6a0-d710-890a1caab5c3')
    else:
        underlayMap(ax, utm, vendor=background)


def makeSymlogTicks(cb, alim):
    """Create symlog ticks for given colorbar."""
    i1 = int(np.log10(alim[0]))
    i2 = int(np.log10(alim[1]))
    ni = i2-i1+1
    lvec = np.linspace(i1, i2, ni)
    lvec = np.append([-10**ele for ele in lvec], 0.)
    ticks = np.append(lvec, [10**ele for ele in np.linspace(i2, i1, ni)])
    cb.set_ticks(ticks)
    cb.set_ticklabels(['{:.0e}'.format(tick) for tick in ticks])


def updatePlotKwargs(**kwargs):
    """Set default values for different plotting tools."""
    kwargs.setdefault("what", "data")
    log = kwargs.setdefault("log", True)
    kwargs.setdefault("color", None)
    kwargs.setdefault("field", 'B')
    kwargs.setdefault("symlog", True)
    if log:
        kwargs.setdefault("cmap", "PuOr_r")
        alim = kwargs.setdefault("alim", [1e-3, 1e0])
    else:
        kwargs.setdefault("cmap", "seismic")
        alim = kwargs.setdefault("alim", [-10., 10.])

    kwargs.setdefault("amphi", False)
    kwargs.setdefault("plim", [-180., 180.])
    llthres = kwargs.setdefault("llthres", alim[0])

    if log and alim[0] != llthres:
        print("Warning, different values vor *llthres* and *alim[0]* are "
              "usually not reasonbale. Continuing ...")

    return kwargs


def makeSubTitles(ax, ncmp, cstr, ci, what):
    """Make subtitles from field type and compontents."""
    for i, ri in enumerate([r'$\Re$($', r'$\Im$($']):
        if what == 'pf':
            ax[i, ncmp].set_title(
                ri + cstr[ci][0] + '_' + cstr[ci][1] + '^p' + '$)')
        elif what == 'sf':
            ax[i, ncmp].set_title(
                ri + cstr[ci][0] + '_' + cstr[ci][1] + '^s' + '$)')
        elif what == 'sf/pf':
            ax[i, ncmp].set_title(
                ri + cstr[ci][0] + '_' + cstr[ci][1] + '^p' + '/' +
                cstr[ci][0] + '_' + cstr[ci][1] + '^s' + '$)')
        else:
            ax[i, ncmp].set_title(ri + cstr[ci][0] + '_' + cstr[ci][1] + '$)')
