"""
Part 4: Gradient Boosted Trees with XGBoost
Run: python part4_xgboost.py

Expected behavior:
- Loads HAR dataset using utils.load_har_dataset()
- Trains an XGBoost multiclass classifier on the training set
- Evaluates on the test set
- Prints test accuracy and displays confusion matrix

"""

import jax
import jax.numpy as jnp
from xgboost import XGBClassifier

from utils import load_har_dataset, display_confusion_matrix_and_accuracy, accuracy


def xgb_train_and_predict(X_train, y_train, X_test, y_test):
    """
    TODO : Train an XGBoost multiclass classifier and return:
      - y_pred (predicted labels for X_test)
      - test_acc (accuracy on the test set, using utils.accuracy)

    Notes:
      - Labels are 1..6, you may shift to 0..5 for training and shift back.
    """
    model = XGBClassifier(objective='multi:softmax', num_class=6, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train - 1)
    y_pred = model.predict(X_test) + 1
    acc = accuracy(y_test, y_pred)
    return y_pred, acc
    


def main():
    print("Part 4: XGBoost (Gradient Boosted Trees)")

    X_train, y_train, X_test, y_test = load_har_dataset()
    y_pred, test_acc = xgb_train_and_predict(X_train, y_train, X_test, y_test)

    print(f"XGBoost Test Accuracy: {float(test_acc)*100:.2f}%")
    display_confusion_matrix_and_accuracy("XGBoost", y_test, y_pred)


if __name__ == "__main__":
    main()
