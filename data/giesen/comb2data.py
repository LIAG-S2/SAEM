import numpy as np
from saem import CSEMData


txpos = np.array([[559497.46, 5784467.953],
                  [559026.532, 5784301.022]]).T
ground = 70  #  altitude of ground
# %% read in the old data and take every second data point (12->24m)
self = CSEMData(datafile="data_f[1-3].mat", txPos=txpos, txalt=ground)
self.showField("alt")
self.filter(every=2)
self.showField("alt")
# %% reading the second part (already 24m distance)
part2 = CSEMData(datafile="newdata_f*.mat", txPos=txpos, txalt=ground)
part2.showField("alt")
self.addData(part2)
self.showField("alt")
# %% rotate to Tx so that tx distances can be used for filtering
nf = 5  # frequency index for checking
self.showData(nf=nf)
# %%
self.simulate(rho=25) # make 0D simulation to check
self.showData(nf=nf, what="response")
dsfsdfsfsdf
# %%
self.rotate()  # according to transmitter
self.showData(nf=nf)
