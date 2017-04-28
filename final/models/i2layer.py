import sys; sys.path.append('../')
from inception import *

cnn = inception()

# =================================
# Load the train and valid data
# =================================
cnn._loadTrain('../data/train/')
cnn._loadValid('../data/valid/')

# =================================
# Build the Model
# =================================
cnn._inception(convl=[3,5,23])
cnn._inception(convl=[3,5,23])
cnn._convl()
cnn._flatten()
cnn._dense(256, 'elu', 0.5, 2)
cnn._compile()
#cnn._plotCNN(to_file='img/i2layer.png')

# =================================
# Train the model
# =================================
cnn._train(250, 32, patience=25, log='log/i2layer.log', weights='weights/i2layer.h5')


