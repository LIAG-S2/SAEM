import numpy as np
# import matplotlib.pyplot as plt
from saem import CSEMData, CSEMSurvey


# %% Patch 1 (parts a and b)
tx1 = np.array([[559497.46, 5784467.953],
                [559026.532, 5784301.022]]).T
data1 = CSEMData('20211004/data*.mat', txPos=tx1)
data1.filter(every=2)
data1.addData('20220124/data*.mat')
# %% patch 2
tx2 = np.array([[559650.46, 5784095.953],
                [559130.532, 5783925.022],
                [559035.532, 5783870.022]]).T
data2 = CSEMData('20220209/data*.mat', txPos=tx2)
# %% filter data
for i, data in enumerate([data1, data2]):
    print(data)
    data.radius = 10
    data.detectLinesAlongAxis("y")
    data.showField("line")
    data.filter(fmin=90, fmax=6000)
    data.filter(minTxDist=100)
    data.estimateError(absError=1e-3, relError=0.05)
    data.deactivateNoisyData(aErr=0.0001, rErr=0.5)
    data.saveData("tx{:d}BxByBz.npz".format(i+1))

# %%
self = CSEMSurvey()
self.addPatch(data1)
self.addPatch(data2)
self.basename='Giesen'
self.saveData()
self.saveData(cmp=[0, 0, 1])
