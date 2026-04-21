import numpy as np

for f in np.linspace(0, 1.0, 1000):
   if np.abs(np.cos(f) - 9 * np.sin(f) - (-0.813634)) < 0.01:
       print("Found f:", f)
