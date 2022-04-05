import numpy as np
from saem import CSEMSurvey
import pyproj

utm = pyproj.Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
origin = [701449.9, 5605079.4]
ang = 47 / 180 * np.pi

tx13 = np.array([[11.91913611977991,50.50155677863203,521.3421840220359],
                 [11.92876525047014,50.5055442323089,521.3421840220359],
                 [11.9314488514447,50.50682826605019,521.3421840220359]]).T
x, y = utm(*tx13[:2])
A = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
tx, ty = np.column_stack((x-origin[0], y-origin[1])).dot(A).T

pos = np.column_stack((tx, ty, tx13[2]))
np.savetxt("Tx13.pos", pos, fmt="%7.2f")
