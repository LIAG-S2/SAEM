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
rho2 = 5
rho = [1000, rho2, 1000]
thk = [100, 100]
self.simulate(rho=rho, thk=thk)
self.basename = "1000 {:d} 1000".format(rho2)
# self.ERR = self.RESP * 0.01
# %%
self.showLineData(what="response")
# %%
ax=None
for i, f in enumerate(self.f):
    ax = self.showLineFreq(line=1, nf=i, what="response", x="x",
                           llthres=1e-3, alim=[1e-3, 10.], ax=ax,
                           label="f = {:d} Hz".format(f), lw=2)
# %%


# for rho2 in [100, 10]:
#     rho = [1000, rho2, 1000]
#     self.simulate(rho=rho, thk=thk)
#     self.showLineFreq(line=1, nf=1, what="response", x="x", ax=ax,
#                       llthres=1e-3, alim=[1e-3, 10.], label=str(rho2))
