from possum import *

spec = possum()
spec._generateParams(N=2000, pcomplex=0.03, seed=7775)
spec._simulateNspec(save=True, outdir='../data/valid/', timeit=True)
