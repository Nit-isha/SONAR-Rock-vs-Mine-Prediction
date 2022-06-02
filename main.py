# Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Data Collection & Processing

data = pd.read_csv("Sonar Data.csv", header=None)
print(data.head())
print(data.shape)
print(data.describe())
print(data[60].value_counts())

X = data.drop(columns=60, axis=1)
Y = data[60]
print(X)
print(Y)

# Split Training and Test Dataset

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=1, test_size=0.3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Model Evaluation

clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Accuracy Score:", accuracy_score(Y_test, Y_pred))

# Prediction for New Value

input_data = (0.0225, 0.0019, 0.0075, 0.0097, 0.0445, 0.0906, 0.0889, 0.0655, 0.1624, 0.1452, 0.1442, 0.0948, 0.0618, 0.1641, 0.0708, 0.0844, 0.2590, 0.2679, 0.3094, 0.4678, 0.5958, 0.7245, 0.8773, 0.9214, 0.9282, 0.9942, 1.0000, 0.9071, 0.8545,
              0.7293, 0.6499, 0.6071, 0.5588, 0.5967, 0.6275, 0.5459, 0.4786, 0.3965, 0.2087, 0.1651, 0.1836, 0.0652, 0.0758, 0.0486, 0.0353, 0.0297, 0.0241, 0.0379, 0.0119, 0.0073, 0.0051, 0.0034, 0.0129, 0.0100, 0.0044, 0.0057, 0.0030, 0.0035, 0.0021, 0.0027)

inputAsArray = np.asarray(input_data)
inputDataReshape = inputAsArray.reshape(1, -1)
prediction = clf.predict(inputDataReshape)

if prediction[0] == "R":
    print("This is a Rock.")
else:
    print("This is a Mine.")
