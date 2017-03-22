from possum import *

spec = possum()
spec._generateParams(N=50000, pcomplex=0.5, seed=10000)
spec._simulateNspec(save=True, outdir='../data/train/')
