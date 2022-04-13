"""Several tools for SAEM."""
import numpy as np

def detectLinesAlongAxis(rx, ry, axis='x', sort=True, show=False):
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

    if sort:
        means = []
        for li in np.unique(dummy):
            if axis == 'x':
                means.append(np.mean(ry[dummy==li], axis=0))
            elif axis == 'y':
                means.append(np.mean(rx[dummy==li], axis=0))
        lsorted = np.argsort(means)
        for li, lold in enumerate(lsorted):
            line[dummy==lold] = li + 1

    return line


def detectLinesByDistance(rx, ry, axis='x', sort=True, show=False,
                          minDist=200.):
    """Split data in lines for line-wise processing."""

    dummy = np.zeros_like(rx, dtype=int)
    line = np.zeros_like(rx, dtype=int)
    li = 0
    for ri in range(1, len(rx)):
        dummy[ri-1] = li
        dist = np.sqrt((rx[ri]-rx[ri-1])**2 +\
                       (ry[ri]-ry[ri-1])**2)
        if dist > minDist:
            li += 1
    dummy[-1] = li

    if sort:
        means = []
        for li in np.unique(dummy):
            if axis == 'x':
                means.append(np.mean(ry[dummy==li], axis=0))
            elif axis == 'y':
                means.append(np.mean(rx[dummy==li], axis=0))
        lsorted = np.argsort(means)
        for li, lold in enumerate(lsorted):
            line[dummy==lold] = li + 1

    return line


def detectLinesOld(rx, ry, show=False):
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
    for i in range(len(sdx)):
        if sdx[i] != 0:
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
