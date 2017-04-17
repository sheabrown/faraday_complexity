from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.1, noiseMax=0.2, pcomplex=0.5, seed=923743)
spec._simulateNspec(save=True, dir='data/train/V2/', timeit=True)
