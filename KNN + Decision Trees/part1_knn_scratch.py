"""
Part 1: KNN from scratch
Run: python part1_knn_scratch.py
This will:
 - load the HAR dataset (expecting the "UCI HAR Dataset" folder in the current dir)
 - run the scratch KNN from util.py for k = 4,6
 - display the confusion matrix and accuracy
"""
import time
from utils import load_har_dataset, ScratchKNN, display_confusion_matrix_and_accuracy

def main():
    print("Part 1: KNN from scratch")
    
    X_train, y_train, X_test, y_test = load_har_dataset()

    for k in [4, 6]:
        knn = ScratchKNN(k=k)

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        display_confusion_matrix_and_accuracy(k, y_test, y_pred)

if __name__ == '__main__':
    main()
