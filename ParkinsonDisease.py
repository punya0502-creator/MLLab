import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")

# (a) Analysis and visualization
print(df.describe())
sns.heatmap(df.drop(columns="name").corr(), cmap="coolwarm")
plt.show()
sns.countplot(data=df, x="status")
plt.show()

# Prepare data
X, y = df.drop(columns=["name", "status"]), df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

# (b) Elbow method
errors = []
for k in range(1, 11):
m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
errors.append(1 - accuracy_score(y_test, m.predict(X_test)))
plt.plot(range(1, 11), errors, marker="o")
plt.xlabel("k")
plt.ylabel("Error rate")
plt.title("Elbow Method")
plt.show()
k = errors.index(min(errors)) + 1
print("Optimal k:", k)

# (c) KNN and Weighted KNN evaluation
for name, weight in [("KNN", "uniform"), ("Weighted KNN", "distance")]:
m = KNeighborsClassifier(n_neighbors=k, weights=weight).fit(X_train, y_train)
pred = m.predict(X_test)
print("\\n", name, "Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, pred))
