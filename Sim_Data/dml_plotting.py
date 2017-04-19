def dml_plotting(x_set,y_set,y_label,x_label,title,plot_name,save,filepath):
	import matplotlib.pyplot as plt
	import numpy as np
	from scipy.stats import kendalltau
	import seaborn as sns
	import pandas as pd
	import time
	version=time.strftime("%m_%d")
	
	plt.ion()	
	
	sns.set(style="ticks")
	X_set = pd.Series(x_set,name=x_label)
	Y_set = pd.Series(y_set,name=y_label)
	

	g = sns.jointplot(X_set,Y_set,kind="hex",size=7, space=0,stat_func=kendalltau, color="#4CB391",gridsize=100)
	
	
	
	g.fig.suptitle(str(title) +" Date:" +version) 
	
	if(save):
		plt.savefig(filepath+plot_name+".png",bbinches='tight')

