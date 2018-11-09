import argparse
import math

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def score_to_bin(Y, interval=0.1):
    # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]

    intervals = np.arange(0, 1.0, interval)
    encoder = LabelEncoder()
    encoder.fit(range(intervals.size))
    Y_cat = []
    for y in Y:
        assert 0 <= y <= 1.0
        label = int(math.floor(y/interval))
        Y_cat.append(label)

    Y_cat = np.array(Y_cat)
    return Y_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', default="feature.TF-IDF.npy")
    parser.add_argument('-l', '--label', default="label.JACCARD.npy")
    parser.add_argument('-i', '--interval', type=float, default=0.1)
    args = parser.parse_args()

    assert args.feature and args.label

    X = np.load(args.feature)
    Y = np.load(args.label)
    Y = score_to_bin(Y, args.interval)
    print(X.shape, Y.shape)
    clf = LogisticRegression()
    clf.fit(X, Y)

    Y_pred = clf.predict(X)
    Y_true = Y

    acc = accuracy_score(Y_true, Y_pred)
    print("acc:", acc)
