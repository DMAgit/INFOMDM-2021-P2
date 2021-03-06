from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import data_preprocessing
from getResults import getResults

# --------------------------
# Naïve Bayes
NB_paramGrid = {'alpha': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]}

NB_gridSearch_model = GridSearchCV(MultinomialNB(), NB_paramGrid, cv=10, n_jobs=-1)
NB_gridSearch_model.fit(data_preprocessing.train, data_preprocessing.y_train)

y_pred = NB_gridSearch_model.predict(data_preprocessing.test)

print("MultinomialNB")
print(f"Best parameters were: {NB_gridSearch_model.best_params_}")
print(f"Train set score: {NB_gridSearch_model.score(data_preprocessing.train, data_preprocessing.y_train)}")
print(f"Test set score: {NB_gridSearch_model.score(data_preprocessing.test, data_preprocessing.y_test)}")

print(metrics.classification_report(data_preprocessing.y_test, y_pred))
print(confusion_matrix(data_preprocessing.y_test, y_pred))

# --------------------------
# Logistic Regression
LR_paramGrid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]}

LR_gridSearch_model = GridSearchCV(LogisticRegression(solver='liblinear'), LR_paramGrid, cv=10, n_jobs=-1)
LR_gridSearch_model.fit(data_preprocessing.train, data_preprocessing.y_train)

y_pred = LR_gridSearch_model.predict(data_preprocessing.test)

print("Logistic Regression")
print(f"Best parameters were: {LR_gridSearch_model.best_params_}")
print(f"Train set score: {LR_gridSearch_model.score(data_preprocessing.train, data_preprocessing.y_train)}")
print(f"Test set score: {LR_gridSearch_model.score(data_preprocessing.test, data_preprocessing.y_test)}")

print(metrics.classification_report(data_preprocessing.y_test, y_pred))
print(confusion_matrix(data_preprocessing.y_test, y_pred))

# --------------------------
# Classification Trees
DT_paramGrid = {'max_depth': [None, 2, 4, 8, 16], 'min_samples_split': [2, 4, 8, 16]}

DT_gridSearch_model = GridSearchCV(DecisionTreeClassifier(), DT_paramGrid, cv=10, n_jobs=-1)
DT_gridSearch_model.fit(data_preprocessing.train, data_preprocessing.y_train)

y_pred = DT_gridSearch_model.predict(data_preprocessing.test)

print("Classification Trees")
print(f"Best parameters were: {DT_gridSearch_model.best_params_}")
print(f"Train set score: {DT_gridSearch_model.score(data_preprocessing.train, data_preprocessing.y_train)}")
print(f"Test set score: {DT_gridSearch_model.score(data_preprocessing.test, data_preprocessing.y_test)}")

print(metrics.classification_report(data_preprocessing.y_test, y_pred))
print(confusion_matrix(data_preprocessing.y_test, y_pred))

# --------------------------
# Random Forests
RF_paramGrid = {'n_estimators': [10, 50, 100, 500, 1000],
                'max_depth': [None, 2, 4, 8, 16], 'min_samples_split': [2, 4, 8, 16]}

RF_gridSearch_model = GridSearchCV(RandomForestClassifier(), RF_paramGrid, cv=10, n_jobs=-1)
RF_gridSearch_model.fit(data_preprocessing.train, data_preprocessing.y_train)

y_pred = RF_gridSearch_model.predict(data_preprocessing.test)

print("Random Forests")
print(f"Best parameters were: {RF_gridSearch_model.best_params_}")
print(f"Train set score: {RF_gridSearch_model.score(data_preprocessing.train, data_preprocessing.y_train)}")
print(f"Test set score: {RF_gridSearch_model.score(data_preprocessing.test, data_preprocessing.y_test)}")

print(metrics.classification_report(data_preprocessing.y_test, y_pred))
print(confusion_matrix(data_preprocessing.y_test, y_pred))

# --------------------------
getResults([NB_gridSearch_model, LR_gridSearch_model, DT_gridSearch_model, RF_gridSearch_model], False, True)
