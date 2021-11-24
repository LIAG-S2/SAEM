from saem.maredata import Mare2dEMData

self = Mare2dEMData("P5.emdata")
print(self)
self.chooseF(fmax=1000)
print(self)
self.chooseF(every=2)
print(self)
print(self.f)
self.basename += "f2"
self.generateDataPDF()
# %%
self.saveData()
for tx in [1, 2]:
    self.saveData(tx=tx)
