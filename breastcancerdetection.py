

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/prawinrajae/Downloads/Machine Learning A-Z (Model Selection)/Classification/Data.csv')
dataset.info()
dataset.head()
print("-*-"*20)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression model")
print(cm)
a= accuracy_score(y_test, y_pred)
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy=",a*100)
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Logistic Regression model K-Fold")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("-*-"*20)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes model")
print(cm)
a= accuracy_score(y_test, y_pred)
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy=",a*100)
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Naive Bayes model K-Fold")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("-*-"*20)


# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("SVM model")
print(cm)
a= accuracy_score(y_test, y_pred)
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy=",a*100)
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("SVM model K-Fold")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("-*-"*20)



# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Decision Tree Classification model")
print(cm)
a= accuracy_score(y_test, y_pred)
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy=",a*100)
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Decision Tree Classification model K-Fold")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("-*-"*20)


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Random Forest Classification model")

print(cm)
a= accuracy_score(y_test, y_pred)
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy=",a*100)
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Random Forest Classification model K-Fold")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print("-*-"*20)
