# Breast Cancer - Random Forest and Extra Trees
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = d.target

# (a) Analysis and visualization
print(X.describe())
sns.boxplot(data=X.iloc[:, :5])
plt.xticks(rotation=45)
plt.show()
sns.heatmap(X.corr(), cmap="coolwarm")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# (b) and (c) Random Forest and Extra Trees
models = {
"Random Forest": RandomForestClassifier(random_state=42),
"Extra Trees": ExtraTreesClassifier(random_state=42)
}
for name, model in models.items():
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("\\n", name)
print("Accuracy:", accuracy_score(y_test, pred))
print("Macro-F1:", f1_score(y_test, pred, average="macro"))
