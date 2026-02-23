"""
Part 3: Scaling KNN with FAISS
Run: python part3_faiss.py
Compares sklearn KNN (for a chosen k) with FAISS L2 search + majority vote.
Note: FAISS may need to be installed. If faiss is not available, the script will explain how to install it.
"""
import time
import jax.numpy as jnp
import faiss
import numpy as np
from utils import load_har_dataset, accuracy, majority_vote, ScratchKNN

def scratch_knn_time_and_acc(X_tr, y_tr, X_ts, y_ts, k=5):
    t0 = time.time()

    model = ScratchKNN(k=k)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ts)
    acc = accuracy(y_ts, y_pred)

    t1 = time.time()
    return (t1 - t0), acc

def build_faiss_index(X_tr):
    Xtr = np.asarray(X_tr, dtype=np.float32)
    d = Xtr.shape[1]

    quantizer = faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFPQ(quantizer, d, 128, 17, 8)
    index.train(Xtr)
    index.add(Xtr)
    index.nprobe = 10
    return index

def faiss_knn_time_and_acc(index, X_ts, y_tr, y_ts, k=5):
    Xts = np.asarray(X_ts, dtype=np.float32)
    ytr = np.asarray(y_tr, dtype=np.int32)

    t0 = time.time()
    D, I = index.search(Xts, k)
    t1 = time.time()

    preds = []
    for i in range(I.shape[0]):
        labs = jnp.asarray(ytr[I[i]], dtype=jnp.int32)
        dists = jnp.asarray(D[i], dtype=jnp.float32)
        preds.append(majority_vote(labs, dists))

    y_pred = jnp.asarray(preds, dtype=jnp.int32)
    acc = accuracy(y_ts, y_pred)
    return (t1 - t0), acc

def main():
    print("Part 3: FAISS scaling comparison")
    X_train, y_train, X_test, y_test = load_har_dataset()

    try:
        index = build_faiss_index(X_train)
    except ImportError:
        index = None
    except Exception as e:
        print("faiss index build failed:", e)
        index = None

    for k in [4, 6, 10, 20]:
        try:
            sk_time, sk_acc = scratch_knn_time_and_acc(X_train, y_train, X_test, y_test, k=k)
            print(f"scratch KNN k={k}: time={sk_time:.3f}s acc={sk_acc*100:.2f}%")
        except Exception as e:
            print("scratch KNN failed:", e)

        if index is None:
            print("faiss not found. Install with e.g. 'pip install faiss-cpu'")
        else:
            try:
                fa_time, fa_acc = faiss_knn_time_and_acc(index, X_test, y_train, y_test, k=k)
                print(f"faiss  k={k}: time={fa_time:.3f}s acc={fa_acc*100:.2f}%")
            except Exception as e:
                print("faiss KNN failed:", e)

        print("-"*40)
if __name__ == '__main__':
    main()