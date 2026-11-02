from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

import numpy as np

example = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = knn.predict(example)
print(f"Predicted class for example {example}: {prediction[0]}")