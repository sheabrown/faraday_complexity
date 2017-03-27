from possum import *

spec = possum()
spec._generateParams(N=6000, pcomplex=0.50, seed=7777)
spec._simulateNspec(save=True, outdir='../data/train/', timeit=True)
