from inception import *

cnn = inception(0.0005)
cnn._loadTrain("data/train/V1/")
cnn._loadValid("data/valid/V1/")
cnn._loadTest("data/test/V1/")

cnn._inception(convl=[3,5,11], pool=[3])
cnn._inception(convl=[3,5,23], pool=[3, 11])
cnn._convl()
cnn._flatten()
cnn._dense(256, 'elu', 0.5, 1)
cnn._compile()
cnn._plotCNN(to_file='graph.png')

cnn._train(15, 32, save=True, weights="weights_V1.h5")
cnn._test(prob=0.5)
print(confusion_matrix(cnn.testLabel_, cnn.testPred_))
