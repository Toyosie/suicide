#importing basic libraries
import numpy as np
import pandas as pd

#reading dataset
dataset = pd.read_csv('data.csv', index_col=0)

#separating into dependent(X) and independent(Y) variables
X = dataset[['gender', 'sexuallity','age','income','race','bodyweight','virgin','friends','social_fear','depressed',
            'job_title','edu_level']]
Y = dataset['attempt_suicide']

#label encoder
from sklearn import preprocessing
encode = preprocessing.LabelEncoder()
col = ['gender', 'sexuallity', 'race', 'bodyweight', 'virgin', 'social_fear', 'depressed', 'job_title', 'edu_level']
X[col] = X[col].apply(encode.fit_transform)

Y = encode.fit_transform(Y)

#Oversampling because data is imbalanced
from imblearn.over_sampling import SMOTE
oversampling = SMOTE()
X,Y = oversampling.fit_resample(X,Y)

#...............................................................................................
#1. Logistic Regression
from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X,Y, test_size=0.1, random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.transform(X_test1)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train1, Y_train1)

Y_pred1 = regressor.predict(X_test1)

print('Logistic Regression...')
from sklearn import metrics
print('Accuracy:',round(metrics.accuracy_score(Y_test1,Y_pred1),2)*100,'%')
print('Precision:',round(metrics.precision_score(Y_test1,Y_pred1),2)*100,'%')
print('F1 score:',round(metrics.f1_score(Y_test1,Y_pred1),2)*100,'%')
print('Cohen Kappa:',round(metrics.cohen_kappa_score(Y_test1,Y_pred1),2)*100, '%')
print('MCC:',round(metrics.matthews_corrcoef(Y_test1,Y_pred1),2)*100, '%')
# print(metrics.classification_report(Y_test1,Y_pred1))
print()

#........................................................................................
#2. Decision Trees
from sklearn.model_selection import train_test_split
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X,Y, test_size=0.1, random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train2)
X_test2 = sc_X.transform(X_test2)

from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC()
classifier.fit(X_train2,Y_train2)

Y_pred2 = classifier.predict(X_test2)

print('Decision Tree...')
from sklearn import metrics
print('Accuracy:',round(metrics.accuracy_score(Y_test2,Y_pred2),2)*100,'%')
print('Precision:',round(metrics.precision_score(Y_test2,Y_pred2),2)*100,'%')
print('F1 score:',round(metrics.f1_score(Y_test2,Y_pred2),2)*100,'%')
print('Cohen Kappa:', round(metrics.cohen_kappa_score(Y_test2,Y_pred2), 2)*100, '%')
print('MCC:', round(metrics.matthews_corrcoef(Y_test2,Y_pred2),2)*100, '%')
# print(metrics.classification_report(Y_test2,Y_pred2))
print()


#..............................................................................................
#3. Random Forests
from sklearn.model_selection import train_test_split
X_train6, X_test6, Y_train6, Y_test6 = train_test_split(X,Y, test_size=0.1, random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train6 = sc_X.fit_transform(X_train6)
X_test6 = sc_X.transform(X_test6)

from sklearn.ensemble import RandomForestClassifier as RFC
forest = RFC()
forest.fit(X_train6, Y_train6)

Y_pred6 = forest.predict(X_test6)

print('Random Forests...')
from sklearn import metrics
print('Accuracy:',round(metrics.accuracy_score(Y_test6,Y_pred6),2)*100,'%')
print('Precision:',round(metrics.precision_score(Y_test6,Y_pred6),2)*100,'%')
print('F1 score:',round(metrics.f1_score(Y_test6,Y_pred6),2)*100,'%')
print('Cohen Kappa:', round(metrics.cohen_kappa_score(Y_test6,Y_pred6), 2)*100, '%')
print('MCC:', round(metrics.matthews_corrcoef(Y_test6,Y_pred6),2)*100, '%')
# print(metrics.classification_report(Y_test6,Y_pred6))
print()


#...................................................................................................
#4. Gradient Boosting
from sklearn.model_selection import train_test_split
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X,Y, test_size=0.1, random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train5 = sc_X.fit_transform(X_train5)
X_test5 = sc_X.transform(X_test5)

from sklearn.ensemble import GradientBoostingClassifier as GBC
boost = GBC()
booster = boost.fit(X_train5, Y_train5)

Y_pred5 = boost.predict(X_test5)

print('Gradient Boosting...')
from sklearn import metrics
print('Accuracy:',round(metrics.accuracy_score(Y_test5,Y_pred5),2)*100,'%')
print('Precision:',round(metrics.precision_score(Y_test5,Y_pred5),2)*100,'%')
print('F1 score:',round(metrics.f1_score(Y_test5,Y_pred5),2)*100,'%')
print('Cohen Kappa:', round(metrics.cohen_kappa_score(Y_test5,Y_pred5), 2)*100, '%')
print('MCC:', round(metrics.matthews_corrcoef(Y_test5,Y_pred5),2)*100, '%')
# print(metrics.classification_report(Y_test5,Y_pred5))
print()


#pickle dump.... to save model
import pickle
pickle.dump(booster, open('booster.pkl', 'wb'))