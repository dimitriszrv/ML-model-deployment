
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


training_data = pd.read_csv("storepurchasedata.csv")

# split the data
X = training_data.iloc[:,:-1].values
y = training_data.iloc[:,-1].values

# divide dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# StandardScaler distributes data
# fit() : used for generating learning model parameters from training data
# transform() : parameters generated from fit() method, applied upon model to generate transformed data set

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance between 2 points
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)

# model training
classifier.fit(X_train, y_train)

# predict the output
y_pred = classifier.predict(X_test)


# predict with probabilities
y_prob = classifier.predict_proba(X_test)[:,1]


# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import pickle
    
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file,'wb'))
scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file,'wb'))