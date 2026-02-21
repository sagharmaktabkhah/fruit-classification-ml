import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import scatter_matrix

# Load dataset
fruits = pd.read_table('fruit.txt')

print(fruits.head())
print(fruits.groupby('fruit_name').size())
print("Dataset shape:", fruits.shape)

# Features and target
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']

# Visualization
cmap = cm.get_cmap('gnuplot')
scatter_matrix(X, c=y, marker='o', s=40,
               hist_kwds={'bins':15},
               figsize=(9,9),
               cmap=cmap)

plt.suptitle('Scatter matrix for each input variable')
plt.savefig('fruits_scatter_matrix.png')
plt.show()

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2
)

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print("Logistic Regression Train Accuracy:", logreg.score(X_train, y_train))
print("Logistic Regression Test Accuracy:", logreg.score(X_test, y_test))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("Decision Tree Train Accuracy:", dt.score(X_train, y_train))
print("Decision Tree Test Accuracy:", dt.score(X_test, y_test))

# KNN
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)

print("KNN Train Accuracy:", kn.score(X_train, y_train))
print("KNN Test Accuracy:", kn.score(X_test, y_test))