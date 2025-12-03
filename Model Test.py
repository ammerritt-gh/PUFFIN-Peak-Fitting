# test_dho_voigt.py
import numpy as np
import matplotlib.pyplot as plt
from models.dho_voigt_model import DhoVoigtComposite

x = np.linspace(-20, 20, 1601)
m = DhoVoigtComposite(gauss_fwhm=1.0, lorentz_fwhm=0.3, bg=5.0, elastic_height=200.0, T=300.0)
# add a couple of phonons: center (meV), height (peak counts), damping
m.add_peak(center=5.0, height=20.0, damping=0.15)
m.add_peak(center=10.0, height=10.0, damping=0.08)

y = m.evaluate(x)

plt.figure(figsize=(8,4))
plt.plot(x, y, label='Model')
plt.axvline(0, color='k', alpha=0.3)
plt.legend()
plt.xlabel('Energy (meV)')
plt.ylabel('Counts')
plt.show()

