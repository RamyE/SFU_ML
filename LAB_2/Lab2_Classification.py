# -*- coding: utf-8 -*-
"""
Simon Fraser University - Mechatronic Systems Engineering
Spring 2021 - MSE491 - Application of Machine Learning in Mechatronic Systems
Lab 2 - Classification
@author: Amin Kabir - kabir@sfu.ca
"""

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# Load General Libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Importing the dataset (no header)
df0 = pd.read_csv("EMG\EMG_0.csv", header=None)
df1 = pd.read_csv("EMG\EMG_1.csv", header=None)
df2 = pd.read_csv("EMG\EMG_2.csv", header=None)
df3 = pd.read_csv("EMG\EMG_3.csv", header=None)
dataset = pd.concat([df0,df1,df2,df3], axis = 0)

# Display What Each Label Refers To
import matplotlib.image as mpimg

img0 = mpimg.imread('EMG/0.jpg')
img1 = mpimg.imread('EMG/1.jpg')
img2 = mpimg.imread('EMG/2.jpg')
img3 = mpimg.imread('EMG/3.jpg')

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img0)
plt.axis('off')
plt.title('0 - rock')
plt.subplot(2, 2, 2)
plt.imshow(img1)
plt.axis('off')
plt.title('1 - scissors')
plt.subplot(2, 2, 3)
plt.imshow(img2)
plt.axis('off')
plt.title('2 - paper')
plt.subplot(2, 2, 4)
plt.imshow(img3)
plt.axis('off')
plt.title('3 - okay')

plt.show()


#%%

# Creating a binary dataset
# Class 0: Closed Fist
# Class 1: Open Hand
dataset_binary = dataset[dataset.iloc[:, -1].isin([0,2])]
dataset_binary.iloc[:, -1].replace(to_replace=2, value=1, inplace=True)
# Split features and targets - X: Features, y: Targets
X_binary = dataset_binary.iloc[:, :-1]
y_binary = dataset_binary.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=0.2)

# Save Test Set
np.savetxt("features_emg_test_binary.csv", X_test, delimiter=",")
np.savetxt("targets_emg_test_binary.csv", y_test, delimiter=",")

# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
X_train_binary = pd.DataFrame(StandardScaler().fit_transform(X_train_binary))
X_test_binary = pd.DataFrame(StandardScaler().fit_transform(X_test_binary))



fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img0)
plt.axis('off')
plt.title('0 - rock')
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('1 - paper')
plt.show()


#%%

# Multi-class dataset
# Class 0: Closed Fist
# Class 1: V sign
# Class 2: Open Hand
# Class 3: OK gesture
# Split features and targets - X: Features, y: Targets
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save Test Set
np.savetxt("features_emg_test.csv", X_test, delimiter=",")
np.savetxt("targets_emg_test.csv", y_test, delimiter=",")

# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))

#%%






#%%


# Some Functions for Showing the Classifier Performance
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

def classifier_performance(model,y_pred):
    print('Classification Report: \n', classification_report(y_test,y_pred))
    # Plot normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     display_labels=['rock','scissors','paper','ok'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    return

### Decision Trees Classification
from sklearn.tree import DecisionTreeClassifier
MODEL_DT = DecisionTreeClassifier()

# Train the Model
MODEL_DT.fit(X_train,y_train)

# Save the Trained Model
pickle.dump(MODEL_DT, open('MODEL_CLASSIFICATION_DT.pkl', 'wb'))

# Evaluate the Trained Model
# Predict the Trained Model on our Test data
y_pred_DT = MODEL_DT.predict(X_test)

# Print the Classification Report and Confusion Matrix
classifier_performance(MODEL_DT,y_pred_DT)


### K Nearest Neighbors Classification
from sklearn.neighbors import KNeighborsClassifier
MODEL_KNN = KNeighborsClassifier(n_neighbors=5)

# Train the Model
MODEL_KNN.fit(X_train,y_train)

# Save the Trained Model
pickle.dump(MODEL_KNN, open('MODEL_CLASSIFICATION_KNN.pkl', 'wb'))

# Evaluate the Trained Model
# Predict the Trained Model on our Test data
y_pred_KNN = MODEL_KNN.predict(X_test)

# Print the Classification Report and Confusion Matrix
classifier_performance(MODEL_KNN,y_pred_KNN)


### Gaussian Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
MODEL_GNB = GaussianNB()

# Train the Model
MODEL_GNB.fit(X_train,y_train)

# Save the Trained Model
pickle.dump(MODEL_GNB, open('MODEL_CLASSIFICATION_GNB.pkl', 'wb'))

# Evaluate the Trained Model
# Predict the Trained Model on our Test data
y_pred_GNB = MODEL_GNB.predict(X_test)

# Print the Classification Report and Confusion Matrix
classifier_performance(MODEL_GNB,y_pred_GNB)


### Support Vector Machines
from sklearn import svm
MODEL_SVM = svm.SVC()

# Train the Model
MODEL_SVM.fit(X_train,y_train)

# Save the Trained Model
pickle.dump(MODEL_SVM, open('MODEL_CLASSIFICATION_SVM.pkl', 'wb'))

# Evaluate the Trained Model
# Predict the Trained Model on our Test data
y_pred_SVM = MODEL_SVM.predict(X_test)

# Print the Classification Report and Confusion Matrix
classifier_performance(MODEL_SVM,y_pred_SVM)