import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = d.target

# (a) Analysis and visualization
print(X.describe())
sns.heatmap(X.corr(), cmap="coolwarm")
plt.show()
sns.countplot(x=y)
plt.show()

# Split, scale and apply PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)
pca = PCA(n_components=0.95)
X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)
print("PCA components:", pca.n_components_)

# (b) and (c) SVM models and evaluation
for name, kernel in [("Linear SVM", "linear"), ("RBF Kernel SVM", "rbf")]:
m = SVC(kernel=kernel, probability=True, random_state=42).fit(X_train, y_train)
pred = m.predict(X_test)
prob = m.predict_proba(X_test)[:, 1]
print("\\n", name)
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
