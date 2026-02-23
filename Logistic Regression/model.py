import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit
from time import perf_counter

jax.config.update("jax_enable_x64", True)

def read_csv_data(filename):
    df = pd.read_csv(filename)
    X = jnp.array(df.iloc[:, :-1].values, dtype=jnp.float64)
    y = jnp.array(df.iloc[:, -1].values, dtype=jnp.float64)
    n = X.shape[0]
    X = jnp.hstack([jnp.ones((n, 1)), X])
    return X, y

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
    alpha = -0.1

    tol = 1e-3
    max_iters = 20000

    gradient_function = jax.grad(loss_function)

    for _ in range(max_iters):
        gradient = gradient_function(beta, standardized, y)
        delta = alpha * gradient
        if jnp.linalg.norm(delta) < tol:
            break
        beta = beta + delta
        iter_count += 1
    
    print(f'Took {iter_count} iterations: {perf_counter() - start :.2f}s')

    return unstandardized(beta, mean, std)

def accuracy(beta, X, y):
    probs = sigmoid(X @ beta)
    preds = (probs >= 0.5).astype(y.dtype)
    return jnp.mean(preds == y)

def model(params, study_hours, attendance_percent, practice_exam_score):
    y_hat = sigmoid(jnp.array([1.0, study_hours, attendance_percent, practice_exam_score]) @ params)
    return 1 if y_hat >= 0.5 else 0
    
def main():
    X, y = read_csv_data("data.csv")
    params = fit(X, y)
    pred = model(params, 15, 100, 100)
    print("Pass" if pred == 1 else "Fail")
    print(f'Accuracy: {accuracy(params, X, y)}')
    prob = sigmoid(jnp.array([1, 15, 100, 100]) @ params)
    print(prob)

main()