import numpy as np
from saem import CSEMSurvey, CSEMData

tx13 = np.loadtxt("Tx13.pos")
marefile = "Schleiz_Tx13_L01_L14_BxByBz_47deg_estmationerrors.emdata"
self = CSEMSurvey(marefile, txs=[tx13], flipxy=True)
self.showData()
p = self.patches[0]
# p.generateDataPDF(marefile.replace(".emdata", ".pdf"))
p.saveData()
bla=CSEMData("patch1BxByBz.npz")  # check load since restructure
