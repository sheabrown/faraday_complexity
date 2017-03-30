from possum import *

spec = possum()
spec._generateParams(N=5000, pcomplex=0.30, seed=5555)
spec._simulateNspec(save=True, outdir='../data/valid/', timeit=True)
