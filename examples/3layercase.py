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
kw = dict(line=1, what="response", x="x", llthres=1e-3, alim=[1e-3, 10.], lw=2)
ax=None
for i, f in enumerate(self.f):
    ax = self.showLineFreq(nf=i, ax=ax, label="f = {:d} Hz".format(f), **kw)
# %%
ax = self.showLineFreq(nf=1, label=str(rho2), **kw)
for rho2 in [100, 10]:
    rho = [1000, rho2, 1000]
    self.simulate(rho=rho, thk=thk)
    self.showLineFreq(nf=1, label=str(rho2), **kw)
# %%
# self.simulate(rho=10000)
# self.showLineData(what="response", amphi=True, alim=[1e-3, 1])
