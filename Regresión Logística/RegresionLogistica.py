# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:11:00 2020

@author: str_aps
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb


dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")
dataframe.head()

dataframe.describe()

print(dataframe.groupby('clase').size())

dataframe.drop(['clase'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase',height=4,vars=["duracion", "paginas","acciones","valor"],kind='reg')


X = np.array(dataframe.drop(['clase'],1))
y = np.array(dataframe['clase'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
print(predictions)[0:5]

model.score(X,y)


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

name = 'Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
model.score(X_train,Y_train)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

X_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]})
model.predict(X_new)
