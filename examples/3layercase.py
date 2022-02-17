import numpy as np
from saem import CSEMData

f = [10, 100, 1000]
x = np.arange(10., 3001, 10)

txLen = 1000  # length of the transmitter
altitude = 50
self = CSEMData(f=f, rx=x, alt=altitude,
                txPos=np.array([[0, 0], [txLen/2, -txLen/2]]))
self.cmp = [1, 0, 1]
print(self)
# self.showPos()
# %%
rho2 = 1000
rho = [1000, rho2, 1000]
thk = [100, 100]
self.simulate(rho=rho, thk=thk)
self.basename = "1000 {:d} 1000".format(rho2)
# self.ERR = self.RESP * 0.01
# %%
self.showLineData(what="response", amphi=True)
# %%
# kw = dict(line=1, what="response", x="x", llthres=1e-3, alim=[1e-3, 10.], lw=2)
kw = dict(line=1, what="response", x="x", alim=[1e-3, 10.], lw=2)
ax=None
for i, f in enumerate(self.f):
    ax = self.showLineFreq(nf=i, ax=ax, label="f = {:d} Hz".format(f), **kw)
# %% per frequency with different resistivities
axAP = [self.showLineFreq(nf=i, label="rho={:d}".format(rho2), **kw, amphi=1)
       for i in range(3)]
axRI = [self.showLineFreq(nf=i, label="rho={:d}".format(rho2), **kw)
       for i in range(3)]
for rho2 in [50, 10]:
    rho = [1000, rho2, 1000]
    self.simulate(rho=rho, thk=thk)
    for i in range(3):
        self.basename = "1000 {:d} 1000, f={:d}Hz".format(rho2, self.f[i])
        self.showLineFreq(nf=0, label="rho={:d}".format(rho2), ax=axRI[i], **kw)
        self.showLineFreq(nf=0, label="rho={:d}".format(rho2), ax=axAP[i],
                          amphi=True, **kw)
# %%
# self.showLineFreq(nf=0, what="response", amphi=True)
