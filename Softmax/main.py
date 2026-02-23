import model
import jax.numpy as jnp

def confusion_matrix(beta, X, Y, k=3):
    probs = model.softmax(X @ beta)
    y_pred = jnp.argmax(probs, axis=1)
    y_true = jnp.argmax(Y, axis=1)

    cm = jnp.zeros((k, k), dtype=jnp.int32)
    cm = cm.at[y_true, y_pred].add(1)

    return cm

def main():
    labels = ["Low Quality", "Medium Quality", "High Quality"]

    X, Y = model.read_csv_data("wine_combined.csv")

    X_train, X_test, Y_train, Y_test = model.train_test_split(X, Y, test_size=0.2, seed=42)

    Beta = model.fit(X_train, Y_train)
    cm = confusion_matrix(Beta, X_test, Y_test)

    print("Train accuracy:", model.accuracy(Beta, X_train, Y_train))
    print("Test accuracy:", model.accuracy(Beta, X_test, Y_test))
    print("Confusion Matrix:")
    print(cm)

main()