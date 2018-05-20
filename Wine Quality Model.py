import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
df = pd.read_csv('winequality-white.csv', sep = ';')

# Here you shuffle the rows of the DataFrame and resetting the index to original
df = df.sample(frac=1).reset_index(drop=True)

#print(df.head())
#print(df.describe())

Y = df.iloc[:, 11]
X = df.iloc[:,:11]

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X, Y, test_size = 0.1))

print("X_train contain = ", X_train.shape, "    and    Y_train contain = ", Y_train.shape)
print("X_test  contain = ", X_test.shape, "     and    Y_test   contain = ", Y_test.shape)


model1 = LogisticRegression()
model1.fit(X_train, Y_train)
print('Logistic Regression Test Score = ' , model1.score(X_test, Y_test))

model2 = DecisionTreeClassifier()
model2.fit(X_train, Y_train)
print('Decision Tree Classifier Test Score = ' , model2.score(X_test, Y_test))

model3 = AdaBoostClassifier()
model3.fit(X_train, Y_train)
print('Ada Boost Classifier Test Score = ' , model3.score(X_test, Y_test))

model4 = RandomForestClassifier()
model4.fit(X_train, Y_train)
print('Random Forest Classifier Test Score = ' , model4.score(X_test, Y_test))

model5 = MLPClassifier()
model5.fit(X_train, Y_train)
print('Random Forest Classifier Test Score = ' , model5.score(X_test, Y_test))