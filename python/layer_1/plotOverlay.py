import numpy as np
import matplotlib.pyplot as plt

linestyle = ['-', '--', '-.']
color = ['dodgerblue', 'darkorange', 'forestgreen']
cs = [3, 7, 17]

for i, [c, col] in enumerate(zip(cs, color)):

	cs = '0' + str(c) if c < 10 else str(c)
	cs = 'c' + cs
	cl = str(c)

	F05 = np.load(cs + "c05/F1.npy")
	F11 = np.load(cs + "c11/F1.npy")
	F23 = np.load(cs + "c23/F1.npy")
	T05 = np.load(cs + "c05/threshold.npy")
	T11 = np.load(cs + "c11/threshold.npy")
	T23 = np.load(cs + "c23/threshold.npy")


	plt.figure(1)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.plot(T05, F05, color=col, linestyle=linestyle[0], label=r'$c \in \{' + cl + ', 5\}$')
	plt.plot(T11, F11, color=col, linestyle=linestyle[1], label=r'$c \in \{' + cl + ', 11\}$')
	plt.plot(T23, F23, color=col, linestyle=linestyle[2], label=r'$c \in \{' + cl + ', 23\}$')


	F05 = np.load(cs + "c05/fpr.npy")
	F11 = np.load(cs + "c11/fpr.npy")
	F23 = np.load(cs + "c23/fpr.npy")
	T05 = np.load(cs + "c05/tpr.npy")
	T11 = np.load(cs + "c11/tpr.npy")
	T23 = np.load(cs + "c23/tpr.npy")


	plt.figure(2)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.plot(F05, T05, color=col, linestyle=linestyle[0], label=r'$c \in \{' + cl + ', 5\}$')
	plt.plot(F11, T11, color=col, linestyle=linestyle[1], label=r'$c \in \{' + cl + ', 11\}$')
	plt.plot(F23, T23, color=col, linestyle=linestyle[2],label=r'$c \in \{' + cl + ', 23\}$')


plt.figure(1)
plt.xlabel(r'$p_\mathrm{threshold}$', fontsize=20)
plt.ylabel(r'$F_{1} \, \rm score$', fontsize=20)
plt.legend(loc='lower left')
plt.xlim(0.3, 1)
plt.tight_layout()
plt.savefig("plots/F1_score_all.pdf")

plt.figure(2)
plt.xlabel(r'$\rm FPR$', fontsize=20)
plt.ylabel(r'$\rm TPR$', fontsize=20)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/ROC_curve_all.pdf')

plt.close('all')
