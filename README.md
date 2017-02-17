# Faraday Complexity
This project is an attempt to create a machine-learning algorithm (binary classifier) that will be able to tell whether a polarized radio spectrum is a simple "Faraday thin" source or a more complex source with multiple components or "Faraday thick". The classifier is a convolutional neural network (faraday_cnn.py), which uses the faraday spectrum as the input feature vector. This is all currently work-in-progress. The work is in collaboration with the POSSUM survey with the Australian Square Kilometre Array Pathfinder (ASKAP). 
Bellow is an example of a simple Faraday thin source produced with create_spectrum.py

![alt tag](https://github.com/sheabrown/faraday_complexity/blob/master/rm_spectrum.png)

Here is a two component source, also created with create_spectrum.py

![alt tag](https://github.com/sheabrown/faraday_complexity/blob/master/far_2comp.png)
