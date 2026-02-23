import model
import jax.numpy as jnp

def confusion_matrix(beta, X, y):
    probs = model.sigmoid(X @ beta)
    preds = (probs >= model.THRESHOLD).astype(jnp.int32)
    y_true = y.astype(jnp.int32)

    cm = jnp.zeros((2, 2), dtype=jnp.int32)
    cm = cm.at[y_true, preds].add(1)

    return cm

def main():
    X, y = model.read_csv_data("data.csv")
    X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.2, seed=42)
    
    Beta = model.fit(X_train, y_train)
    cm = confusion_matrix(Beta, X_test, y_test)

    print("Train accuracy:", model.accuracy(Beta, X_train, y_train))
    print("Test accuracy:", model.accuracy(Beta, X_test, y_test))
    print("Confusion Matrix:")
    print(cm)

    print("=========================================================")

    student = [5, 50, 70]
    label, p = model.predict(Beta, student[0], student[1], student[2])

    print(f"Study hours: {student[0]}")
    print(f"Lecture attendance: {student[1]}%")
    print(f"Practice exam score: {student[2]}%")
    print(f"Prediction: {label}")
    print(f"Probability: {p * 100:.2f}%")

    print("=========================================================")



main()