import numpy as np
from saem import CSEMSurvey

tx13 = np.loadtxt("Tx13.pos")
marefile = "Schleiz_Tx13_L01_L14_BxByBz_47deg_estmationerrors.emdata"
self = CSEMSurvey(marefile, txs=[tx13], flipxy=True)
self.showData()
# self.basename=self.patches[0].basename
self.patches[0].generateDataPDF(marefile.replace(".emdata", ".pdf"))
