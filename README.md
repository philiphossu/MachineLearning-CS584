# MachineLearning-CS584
Coursework from CS584, Machine Learning @ IIT

## 1. Programming Assignment 1 (HW1)

Topic(s): Decision Trees, Accuracy, Scikit-Learn. 

Task(s): Listed in the code file. Create a test/train split for each dataset using part of student ID as the random state. For depths d ranging from 1 to 15, create decision trees of max depth d, fit the classifier to the training data, and compute accuracy on test/train splits. Record results and discuss observations.

## 2. Programming Assignment 2 (HW2)

Topic(s): Naive Bayes, Parameter Estimation, Joint Log Likelihood, Scikit-Learn. 

Task(s): Load the datasets and create a test/train split for each using part of student ID as the random state. For 15 different values of alpha ranging from 10^-6 to 10^7: (1) create a Bernoulli Naive Bayes model with that alpha, (2) fit the model to the training set, (3) Compute JLL for training set and store it, (4) Compute JLL for testing set and store it. Store results, pickle results, discuss. 

## 3. Programming Assignment 3 (HW3)

Topic(s): Logistic regression, L1 vs L2 Regularization, Scikit-Learn

Task(s): Load the datasets from HW2 and create test/train split for each same as before. For 15 different C values from 10^-7  to 10^7: (1) Create L2 regularized logistic regression classifier, (2) fit the model to the training set, (3) compute model complexity and store it, (4) compute number of 0 weights, store it, (5) compute CLL for train set, store it, (6) compute CLL for test set, store it, (7) Create L1 regularized logistic regression model, (8) fit to training data, (9) compute number of 0 weights, store it.  

## 4. Final Project (Final_Project)

Topic(s): K-Nearest Neighbors, Support Vector Machines, Principal Components Analysis, Confusion Matrices, F1, Neural Networks, Convolutional Neural Networks, Transfer Learning, Keras, Tensorflow.

Task(s): Detailed in the project_writeup.pdf. 

File(s):

| File      | Description |
| ----------- | ----------- |
| ProjectFile.ipynb      | Main python code notebook for the project      |
| project_writeup.pdf   | Information about the project requirements, phases, and results        |
| bush_test_66.model | Phase 3 Bush CNN|
|williams_test_94.model| Phase 3 Williams CNN|
|bush.model| Phase 4 Bush CNN|
|williams.model| Phase 4 Williams CNN|
|yaledata_cleaned_pickled| Phase 4 Transfer Learning Data|

Notes:
- The primary data files were too large to upload, happy to provide a link. 
- A *lot* of trial and error went into the testing of various models. The code here only shows the final results. 
- The code in the ProjectFile.ipynb file was written over the course of several weeks on multiple machines. It may not work perfectly out the box depending on the Python, Keras, and TensorFlow versions. 
