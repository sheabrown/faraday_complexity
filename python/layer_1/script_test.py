import sys
sys.path.append('../')
from inception import *

if __name__ == '__main__':
	try:
		c1 = int(sys.argv[1])
		c2 = int(sys.argv[2])
	except:
		print('Abort')
		sys.exit(1)

	p = 3
	nb = 15

	# Load the data
	dir = '../../data/'
	cnn = inception()
	cnn._loadTrain(dir + "train/X_data.npy", dir + "train/label.npy")
	cnn._loadValid(dir + "valid/X_data.npy", dir + "valid/label.npy")
	cnn._loadTest( dir + "test/X_data.npy",  dir + "test/label.npy")

	# Extract the directory info and make if it doesn't exist
	c1s = '0' + str(c1) if c1 < 10 else str(c1)
	c2s = '0' + str(c2) if c2 < 10 else str(c2)
	dir = 'c' + c1s + 'c' + c2s + '/'

	if not os.path.exists(dir):
	    os.makedirs(dir)

	# Create the model
	cnn._convl1D()
	cnn._inception(convl=[c1, c2], pool=[p])
	cnn._convl1D()
	cnn._flatten()
	cnn._dense(512, 'relu', 0.5, 2)
	cnn._compile(2, 'softmax', 'adadelta', 'binary_crossentropy', ['binary_accuracy'])

	# Train and test the model
	cnn._train(nb, 5, save=True, ofile=dir + "wtf_weights")
	cnn._test(prob=0.5)

	# Get the F1 and ROC data and save
	cnn._getF1(save=True, dir=dir)
	cnn._getROC(save=True, dir=dir)

	cnn._test(prob=0.7)
	print(confusion_matrix(cnn.testLabel_, cnn.testPred_))
