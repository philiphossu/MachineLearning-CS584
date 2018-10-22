import pandas as pd
import pickle
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Load Datasets
print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

# Setup alphas list for BernoulliNB parameter
alphas = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5,10**6,10**7]

# Setup data structures to hold train, test results
train_jll = np.zeros((10, 15))
test_jll = np.zeros((10, 15))

for i in range(0,10):
    idx = 0
    # Split datasets
    x_train, x_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=7000)
    for j in alphas:
        # 1. Create new Bernoulli Naive Bayes model using alpha value
        mod = BernoulliNB(alpha=j)
        # Fit the model to the training set
        mod.fit(x_train,y_train)
        # Compute the joint log likelihood for the training set, store it train_jll 2d array
        total_res = mod._joint_log_likelihood(x_train)
        y_train_binary = y_train*1
        entry_val = 0
        # Sum-up by matching true labels
        for k in range(0,len(y_train)):
            entry_val += total_res[k][y_train_binary[k]]
        # Store result
        train_jll[i][idx] = entry_val
        # 2. Compute the joint log likelihood for the testing set, store it test_jll 2d array
        total_res = mod._joint_log_likelihood(x_test)
        y_test_binary = y_test*1
        entry_val = 0
        # Sum-up by matching true labels
        for k in range(0,len(y_test)):
            entry_val += total_res[k][y_test_binary[k]]
        test_jll[i][idx] = entry_val
        idx += 1


pickle.dump((train_jll, test_jll), open('results.pkl', 'wb'))
