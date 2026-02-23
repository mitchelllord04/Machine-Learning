import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad, jit
from jax import Array
from jax.typing import ArrayLike
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# Read in the CSV file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_csv_data(infilename):
    df = pd.read_csv(infilename, index_col=0)
    n, d = df.values.shape
    d = d - 1
    X = jnp.array(df.values[:, :-1], dtype=jnp.float64)
    labels = list(df.columns[:-1])
    y = jnp.array(df.values[:, -1], dtype=jnp.float64)
    X = jnp.hstack([jnp.ones((n, 1), dtype=jnp.float64), X])
    return X, y, labels

# Returns a vector of weights
def matrix_inversion_fit(X: ArrayLike, y: ArrayLike) -> Array:
    return jnp.linalg.solve(X.T @ X, X.T @ y)

# Helper function to standardize data
def standardize(X: ArrayLike) -> tuple[Array, Array, Array]:
    mean = jnp.mean(X[:, 1:], axis=0)
    std_dev = jnp.std(X[:, 1:], axis=0)
    X_std_features = (X[:, 1:] - mean) / std_dev
    X_standardized = jnp.concatenate([X[:, :1], X_std_features], axis=1)
    return X_standardized, mean, std_dev

# Helper function to un-standardize coefficients
def params_for_unstandardized(beta: ArrayLike, means: ArrayLike, std: ArrayLike) -> Array:
    slopes = beta[1:] / std
    intercept = beta[0] - jnp.sum((beta[1:] * means) / std)
    return jnp.concatenate([jnp.array([intercept]), slopes])

# Returns a vector of weights
def gradient_descent_fit(X: ArrayLike, y: ArrayLike) -> Array:
    # Standardize the data (except for the first column!)
    # Get the mean and standard deviation for each column

    standardized, means, std_devs = standardize(X)

    # Iteratively use gradient descent to get
    # a good estimate of the parameters
    # Start with all zeros as your first guess
    # Stop when the change in the parameters gets very small
    # Experiment to find a good learning rate
    # You should expect a few hundred iterations
    iter_count = 0
    start = perf_counter()

    beta = jnp.zeros(standardized.shape[1])
    alpha = -0.1
    tol = 1e-6
    max_iters = 20000

    n = len(y)

    for _ in range(max_iters):
        gradient = (-2.0 / n) * (standardized.T @ (y - standardized @ beta))
        delta = alpha * gradient
        if jnp.linalg.norm(delta) < tol:
            break
        beta = beta + delta
        iter_count += 1
    


    print(f"Took {iter_count} iterations to converge: {perf_counter() - start:.6f} seconds")

    # Un-standardize the coefficients before you return them
    return params_for_unstandardized(beta, means, std_devs)


# Helper function for autodiff
@jax.jit
def loss_function(beta: ArrayLike, X: ArrayLike, y: ArrayLike) -> float:
    return jnp.mean((y - X @ beta) ** 2)

# Returns a vector of weights using JAX autodiff
def autodiff_descent_fit(X: ArrayLike, y: ArrayLike) -> Array:
    # Standardize the data
    standardized, means, std_devs = standardize(X)

    # Have JAX create a function that returns the gradient of the loss function
    # gradient_function = grad(loss_function)
    # Iteratively use gradient descent
    iter_count = 0
    start = perf_counter()

    beta = jnp.zeros(standardized.shape[1])
    alpha = -0.1
    tol = 1e-6
    max_iters = 20000

    gradient_function = jax.grad(loss_function)

    for _ in range(max_iters):
        gradient = gradient_function(beta, standardized, y)
        delta = alpha * gradient
        if jnp.linalg.norm(delta) < tol:
            break
        beta = beta + delta
        iter_count += 1

    print(f"Took {iter_count} iterations to converge: {perf_counter() - start:.6f} seconds")

    # Un-standardize
    return params_for_unstandardized(beta, means, std_devs)

# Make it pretty
def format_prediction(beta, labels):
    str = f"predicted price = ${beta[0]:,.2f} + "
    d = len(labels)
    for i in range(d):
        b = beta[i + 1]
        label = labels[i]
        str += f"(${b:,.2f} x {label})"
        if i < d - 1:
            str += " + "
    return str

# Return the R2 score for coefficients B
# Given inputs X and outputs y
def score(beta, X, y):
    return 1 - (jnp.sum((y - X @ beta) ** 2) / jnp.sum((y - jnp.mean(y)) ** 2))