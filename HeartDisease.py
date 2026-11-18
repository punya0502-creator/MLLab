import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, names=cols, na_values="?").dropna()
df["target"] = df["target"].astype(int)

# (a) Analysis and visualization
print(df.describe())
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()
sns.countplot(data=df, x="target")
plt.show()

# Prepare data
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

# (b) and (c) Models and evaluation
models = [
("Logistic Regression", LogisticRegression(max_iter=1000),
(y_train > 0).astype(int), (y_test > 0).astype(int)),
("Multinomial Logistic Regression", LogisticRegression(solver="lbfgs", max_iter=1000),
y_train, y_test)
]
for name, model, yt, yv in models:
model.fit(X_train, yt)
pred, prob = model.predict(X_test), model.predict_proba(X_test)
auc = roc_auc_score(yv, prob[:, 1]) if len(model.classes_) == 2 else roc_auc_score(yv, prob, multi_class="ovr", average="weighted")
print("\\n", name)
print("Accuracy:", accuracy_score(yv, pred))
print("Precision:", precision_score(yv, pred, average="weighted", zero_division=0))
print("Recall:", recall_score(yv, pred, average="weighted", zero_division=0))
print("F1-score:", f1_score(yv, pred, average="weighted", zero_division=0))
print("ROC-AUC:", auc)
