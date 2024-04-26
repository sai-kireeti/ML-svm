
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].apply(lambda x: iris.target_names[x])

print(iris_df.describe())  
pd.plotting.scatter_matrix(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']],
                           c=iris_df['target'], alpha=0.8, figsize=(12, 10))
plt.savefig("Graph1.png")
plt.show()

iris_df.hist(figsize=(10, 8))
plt.savefig("Graph2.png")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# We'll use an SVM with a 'sigmoid' kernel for this example other kernels like 'poly', 'rbf', 'linear'.
model = SVC(kernel='sigmoid')
model.fit(X_train, y_train)

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    cv_scores.append(accuracy_score(y_test_fold, y_pred))

# Performance metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Cross-Validation Scores:", cv_scores)
