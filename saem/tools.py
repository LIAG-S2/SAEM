"""Several tools for SAEM."""
import numpy as np


def distToTx(rx, ry, tx, ty):
    """Compute minimum distance to segmented Tx."""
    dist2 = np.ones_like(rx) * 1e9
    for i in range(len(tx)-1):
        px = rx - tx[i]
        py = ry - ty[i]
        bx = tx[i + 1] - tx[i]
        by = ty[i + 1] - ty[i]
        t = (px * bx + by * py) / (bx**2 + by**2)
        t = np.maximum(np.minimum(t, 1), 0)
        dist2 = np.minimum(dist2, (px-t*bx)**2+(py-t*by)**2)

    return np.sqrt(dist2)


def detectLinesAlongAxis(rx, ry, axis='x'):
    """Alernative - Split data in lines for line-wise processing."""

    if axis == 'x':
        r = rx
    elif axis == 'y':
        r = ry
    else:
        print('Choose either *x* or *y* axis. Aborting this method ...')
        return

    dummy = np.zeros_like(rx, dtype=int)
    line = np.zeros_like(rx, dtype=int)
    li = 0
    last_sign = np.sign(r[1] - r[0])
    for ri in range(1, len(rx)):
        sign = np.sign(r[ri] - r[ri-1])
        dummy[ri-1] = li
        if sign != last_sign:
            li += 1
            last_sign *= -1
    dummy[-1] = li

    return sortLines(rx, ry, line, dummy, axis)


def detectLinesByDistance(rx, ry, minDist=200., axis='x'):
    """Split data in lines for line-wise processing."""

    dummy = np.zeros_like(rx, dtype=int)
    line = np.zeros_like(rx, dtype=int)
    li = 0
    for ri in range(1, len(rx)):
        dummy[ri-1] = li
        dist = np.sqrt((rx[ri]-rx[ri-1])**2 +
                       (ry[ri]-ry[ri-1])**2)
        if dist > minDist:
            li += 1
    dummy[-1] = li

    return sortLines(rx, ry, line, dummy, axis)


def detectLinesBySpacing(rx, ry, vec, axis='x'):
    """Detect line changes by jumps in point spacing."""
    if axis == 'x':
        r = rx
    elif axis == 'y':
        r = ry
    else:
        print('Choose either *x* or *y* axis. Aborting this method ...')
        return

    return np.argmin(np.abs(np.tile(r, (len(vec), 1)).T - vec), axis=1)


def detectLinesOld(rx, ry):
    """Split data in lines for line-wise processing."""
    dt = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)
    dtmin = np.median(dt) * 2
    dx = np.round(np.diff(rx) / dt * 2)
    dy = np.round(np.diff(ry) / dt * 2)
    sdx = np.hstack((0, np.diff(np.sign(dx)), 0))
    sdy = np.hstack((0, np.diff(np.sign(dy)), 0))
    line = np.zeros_like(rx, dtype=int)
    nLine = 1
    act = True
    for i, sdxi in enumerate(sdx):
        if sdxi != 0:
            act = not act
            if act:
                nLine += 1

        if sdy[i] != 0:
            act = not act
            if act:
                nLine += 1

        if i > 0 and dt[i-1] > dtmin:
            act = True
            nLine += 1

        if act:
            line[i] = nLine

    return line


def sortLines(rx, ry, line, dummy, axis):
    """Sort line elements by Rx or Ry coordinates."""
    means = []
    for li in np.unique(dummy):
        if axis == 'x':
            means.append(np.mean(ry[dummy==li], axis=0))
        elif axis == 'y':
            means.append(np.mean(rx[dummy==li], axis=0))

    lsorted = np.argsort(means)
    for li, lold in enumerate(lsorted):
        line[dummy == lold] = li + 1

    return line

def readCoordsFromKML(xmlfile, proj='utm', zone=32, ellps="WGS84"):
    """Read coordinates from KML file.

    Parameters
    ----------
    xmlfile : str
        XML or KML file
    proj : str ['utm']
        projection (UTM)
    zone : int [32]
        UTM zone
    ellps : str ['WGS84']
        ellipsoid

    Returns
    -------
    pos : np.array (Nx3)
        matrix of x, y, z positions
    """
    import pyproj
    import xml.etree.ElementTree as ET

    utm = pyproj.Proj(proj=proj, zone=zone, ellps=ellps)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    X, Y, Z = [], [], []
    for line in root.iter("*"):
        if line.tag.find("coordinates") >= 0:
            try:
                lin = line.text.replace("\n", "").replace("\t", "")
            except AttributeError:
                lin = root[0][4][2][1].text.replace("\n", "").replace("\t", "")
            for col in lin.split(" "):
                if col.find(",") > 0:
                    vals = np.array(col.split(","), dtype=float)
                    if len(vals) > 2:
                        X.append(vals[0])
                        Y.append(vals[1])
                        Z.append(vals[2])

    return np.vstack((*utm(X, Y), Z))


def coverage(inv, invmodel=None):
    """ Calculate coverage from inversion instance."""
    if invmodel is None:
        invmodel = inv.model
    if hasattr(inv.fop, '_jac'):
        # for single forward operator
        cov = np.zeros(inv.fop._jac.cols())

        dataScale = inv.dataTrans.deriv(inv.fop.last_response) / \
            inv.dataTrans.error(inv.fop.last_response, inv.fop.errors)
        for i in range(inv.fop._jac.rows()):
            cov += np.abs(inv.fop._jac.row(i) * dataScale[i])
        cov /= inv.modelTrans.deriv(invmodel) # previous * invmodel
        cov /= inv.fop.mesh().cellSizes()
    else:
        # for joint modelling
        cov = np.zeros(inv.fop.fops[0]._jac.cols())
        for n in range(len(inv.fop.fops)):
            dataScale = inv.dataTrans.deriv(inv.fop.fops[n].last_response) / \
                inv.dataTrans.error(inv.fop.fops[n].last_response,
                                    inv.fop.fops[n].errors)
            for i in range(inv.fop.fops[n]._jac.rows()):
                cov += np.abs(inv.fop.fops[n]._jac.row(i) * dataScale[i])
        cov /= inv.modelTrans.deriv(invmodel) # previous * invmodel
        cov /= inv.fop.fops[0].mesh().cellSizes()

    return cov

def isPointInsidePolygon(x, y, polygon):
    """Determine whether point (x, y) is inside a closed polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside
