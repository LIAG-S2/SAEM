import numpy as np
from saem import CSEMSurvey, CSEMData, Mare2dEMData

tx11 = np.loadtxt("tx11.pos")
tx12 = np.loadtxt("tx12.pos")
if 1:  # way 1: initialize by reading Mare data and process patches inplace
    marefile = "Schleiz_Tx11_Tx12_L01_L13_BxByBz_47deg_estmationerrors.emdata"
    if 1:  # way 1b: straight Tx from names in emdata or krooked from files
        self = CSEMSurvey(marefile, flipxy=True, txs=[tx12, tx11])
    else:
        mare = Mare2dEMData(marefile)
        # Tx=1 is actually 2 and vice versa (MB), you could change this here
        self = CSEMSurvey()
        ntx = len(mare.txPositions())
        txs = [np.loadtxt("tx{:d}.pos".format(i+1)) for i in range(ntx)]
        if 0:  # the ugly way: saving and loading files
            for i in range(ntx):
                mare.saveData(tx=i+1, fname="tmp")
                patch = CSEMData("tmp{:d}.npz".format(i+1))
                patch.tx = txs[i][:, 0]
                patch.ty = txs[i][:, 1]
                self.addPatch(patch)
        else:  # the nice way
            self.importMareData(mare, [tx2, tx1])
else:  # way 2: initialize empty survey and add patches
    self = CSEMSurvey()
    if 1:  # way 2a: read in data, filter and add instances
        tx1 = CSEMData("schleiz-Tx1new2BxBz.npz")
        # tx1.filter...
        tx1.saveData("bla")
        tx2 = CSEMData("schleiz-Tx2new2BxBz.npz")
        self.addPatch(tx1)
        self.addPatch(tx2)
    else:  # way 2b: add readily processed numpy data files
        self.addPatch("schleiz-Tx1new2BxBz.npz")
        self.addPatch("schleiz-Tx2new2BxBz.npz")
        # could possibly also work with self.addPatch("data*.mat")

print(self)
fig, ax = self.showPositions()
fig.savefig("pos.pdf")
# %%
self.patches[0].generateDataPDF("Tx12data.pdf")
self.patches[1].generateDataPDF("Tx11data.pdf")
# p.filter()...
# %%
self.showData(nf=8)
self.saveData(cmp=[1, 0, 1])
