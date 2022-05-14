import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.linalg import norm

from pytntnn import tntnn

print("Create a NNLS problem")
# Generate some random data
np.random.seed(42)

n_samples = 200
n_features = 50
X = np.random.randn(n_samples, n_features)
true_coef = 3 * np.random.randn(n_features)
# Threshold coefficients to render them non-negative
true_coef[true_coef < 0] = 0
y = np.dot(X, true_coef)

# Add some noise
y += 5 * np.random.normal(size=(n_samples,))

# Split the data in train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

print(f"{X_train.shape = }")
print(f"{y_train.shape = }")
print()

# Fit the Non-Negative least squares using sklearn
print("Solve using sklearn")
reg_nnls = LinearRegression(positive=True)
y_pred_sklearn = reg_nnls.fit(X_train, y_train).predict(X_test)
r2_score_sklearn = r2_score(y_test, y_pred_sklearn)
print(f"{r2_score_sklearn = }")

sklearn_coef = reg_nnls.coef_
print(f"{norm(sklearn_coef - true_coef) = }")
print()

# Fit the Non-Negative least squares using TNT-NNLS
print("Solve using TNT-NNLS method")

tnt_result = tntnn(X_train, y_train, verbose=True)
tnt_coef = tnt_result.x

print(f"{tnt_result.OuterLoop = }")
print(f"{tnt_result.TotalInnerLoops = }")
print()

y_pred_tnt = X_test @ tnt_coef
r2_score_tnt = r2_score(y_test, y_pred_tnt)
print(f"{r2_score_tnt = }")

print(f"{norm(tnt_coef - true_coef) = }")
