import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

# (a) Analysis and visualization
print(df.describe())
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()
sns.scatterplot(data=df, x="alcohol", y="quality")
plt.show()

# (b) Regression models
X, y = df.drop("quality", axis=1), df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

# (c) Evaluation
for name, model in [("Linear Regression", LinearRegression()), ("Ridge Regression", Ridge(alpha=1))]:
model.fit(X_train, y_train)
p = model.predict(X_test)
print("\\n", name)
print("R2:", r2_score(y_test, p))
print("RMSE:", mean_squared_error(y_test, p)**0.5)
print("MAE:", mean_absolute_error(y_test, p))
