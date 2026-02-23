import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit
from time import perf_counter

jax.config.update("jax_enable_x64", True)

THRESHOLD = 0.62

def read_csv_data(filename):
    df = pd.read_csv(filename)
    X = jnp.array(df.iloc[:, :-1].values, dtype=jnp.float64)
    y = jnp.array(df.iloc[:, -1].values, dtype=jnp.float64)
    n = X.shape[0]
    X = jnp.hstack([jnp.ones((n, 1)), X])
    return X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    n = X.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    test_n = int(n * test_size)
    test_idx = perm[:test_n]
    train_idx = perm[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X):
    mean = jnp.mean(X[ :, 1: ], axis = 0)
    std = jnp.std(X[ :, 1: ], axis = 0)
    X_std_features = (X[ :, 1: ] - mean) / std
    X_standardized = jnp.concatenate([X[ :, :1 ], X_std_features], axis=1)
    return X_standardized, mean, std

def sigmoid(x):
    return 1 / (1 + jnp.exp(-1 * x))

@jax.jit
def loss_function(beta, X, y):
    y_hat = sigmoid(X @ beta)
    eps = 1e-7
    y_hat = jnp.clip(y_hat, eps, 1 - eps)
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))

def unstandardized(beta, means, std):
    slopes = beta[1:] / std
    intercept = beta[0] - jnp.sum((beta[1:] * means) / std)
    return jnp.concatenate([jnp.array([intercept]), slopes])

def fit(X, y):
    standardized, mean, std = standardize(X)

    iter_count = 0
    start = perf_counter()

    beta = jnp.zeros(standardized.shape[1])
    alpha = 0.1

    tol = 1e-3
    max_iters = 20000

    gradient_function = jax.grad(loss_function)

    for _ in range(max_iters):
        gradient = gradient_function(beta, standardized, y)
        delta = alpha * gradient
        if jnp.linalg.norm(delta) < tol:
            break
        beta = beta - delta
        iter_count += 1
    
    print(f'Took {iter_count} iterations: {perf_counter() - start :.2f}s')

    return unstandardized(beta, mean, std)

def accuracy(beta, X, y, threshold=THRESHOLD):
    probs = sigmoid(X @ beta)
    preds = (probs >= threshold).astype(y.dtype)
    return jnp.mean(preds == y)

def predict(params, study_hours, attendance_percent, practice_exam_score, threshold=THRESHOLD):
    x = jnp.array([1.0, study_hours, attendance_percent, practice_exam_score], dtype=jnp.float64)
    p = sigmoid(x @ params)
    return ("Pass" if p >= threshold else "Fail"), float(p)