import pandas as pd
import jax
from jax import jit
import jax.numpy as jnp
from time import perf_counter

jax.config.update("jax_enable_x64", True)

def read_csv_data(filename):
    df = pd.read_csv(filename)

    y_raw = df["quality"].values
    X = df.drop("quality", axis=1).values

    y = jnp.where(y_raw <= 4, 0,
         jnp.where(y_raw <= 6, 1, 2))

    X = jnp.array(X, dtype=jnp.float64)
    y = jnp.array(y, dtype=jnp.int32)

    n = X.shape[0]
    X = jnp.hstack([jnp.ones((n, 1)), X])

    Y = jnp.eye(3)[y]

    return X, Y

def train_test_split(X, Y, test_size=0.2, seed=42):
    n = X.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    test_n = int(n * test_size)
    test_idx = perm[:test_n]
    train_idx = perm[test_n:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

def accuracy(beta, X, Y):
    probs = softmax(X @ beta)
    y_pred = jnp.argmax(probs, axis=1)
    y_true = jnp.argmax(Y, axis=1)
    return float(jnp.mean(y_pred == y_true))

def softmax(Z):
    Z = Z - jnp.max(Z, axis=1, keepdims=True)
    exp = jnp.exp(Z)
    return exp / jnp.sum(exp, axis=1, keepdims=True)


def standardize(X):
    mean = jnp.mean(X[ :, 1: ], axis = 0)
    std = jnp.std(X[ :, 1: ], axis = 0)
    X_std_features = (X[ :, 1: ] - mean) / std
    X_standardized = jnp.concatenate([X[ :, :1 ], X_std_features], axis=1)
    return X_standardized, mean, std

def unstandardize_beta(beta, mean, std):
    beta_0 = beta[0, :]
    beta_rest = beta[1:, :]

    new_beta_rest = beta_rest / std[:, None]

    new_beta_0 = beta_0 - jnp.sum((beta_rest * mean[:, None]) / std[:, None], axis=0)

    new_beta = jnp.vstack([new_beta_0, new_beta_rest])

    return new_beta

def fit(X, Y):
    standardized, mu, sigma = standardize(X)

    alpha = 0.12
    tol = 1e-5
    beta = jnp.zeros((standardized.shape[1], 3), dtype=jnp.float64)
    max_iters = 25000
    n = standardized.shape[0]
    iter_count = 0
    start = perf_counter()

    for _ in range(max_iters):
        gradient = (standardized.T @ (softmax(standardized @ beta) - Y)) / n
        delta = alpha * gradient
        if jnp.linalg.norm(delta) < tol:
            break
        beta = beta - delta
        iter_count += 1
    
    print(f'Took {iter_count} iteration: {perf_counter() - start :.2f}s')

    return unstandardize_beta(beta, mu, sigma)


