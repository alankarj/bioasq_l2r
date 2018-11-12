""" Offline Supervised Model Train and Testing
Usage:
    offline.py train --model=<str> [options]
    offline.py test --model-path=<file> [options]
    offline.py example --model-path=<file> [options]

Options:
    --data-dir=<file>       Directory to data [default: ./data/summary]
    --feature=<str>         Feature type to use [default: TF-IDF]
    --label=<str>           Label type to use [default: JACCARD]
    --interval=<float>      Interval for binary classification [default: 0.1]
    --model=<str>           Model type to use [default: LinearSVC]
    --seed=<int>            Seed number for random generator [default: 11731]
    --save-dir=<file>       Directory to save trained model [default: ./save]
    --model-path=<file>     Path to model pickle [default: ./save/LinearSVC_TF-IDF_JACCARD_0.1.pickle]
"""
import argparse
import math
import json
import os
import pickle
import random

import numpy as np
import numpy as np
from docopt import docopt
from scipy.sparse import hstack

from deiis.model import DataSet, Serializer
from Featurizer import vectorize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def score_to_bin(Y, interval=0.1):
    """
    Maps scores to category, and also returns a dictionary
    """
    # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]
    cat2score = {}
    intervals = np.arange(0, 1.0, interval)
    for idx, score in enumerate(intervals):
        cat2score[idx] = score

    Y_cat = []
    for y in Y:
        assert 0 <= y <= 1.0
        label = int(math.floor(y/interval))
        if y == 1:
            label = len(intervals)-1
        Y_cat.append(label)

    Y_cat = np.array(Y_cat)
    return Y_cat, cat2score


def evaluate(clf, X, Y):
    Y_pred = clf.predict(X)
    acc = accuracy_score(Y, Y_pred)
    return acc


class ModelWrapper(object):
    """
    Includes
    - featurizer
    - classifier
    - label to score dict
    """

    def __init__(self, featurizers, clf, cat2score):
        self.featurizers = featurizers
        self.clf = clf
        self.cat2score = cat2score

    def score(self, question, sentence):

        # Featurizer
        question_featurizer, sentence_featurizer = self.featurizers

        question_feature = question_featurizer.transform([question])
        sentence_feature = sentence_featurizer.transform([sentence])

        feature = hstack([question_feature, sentence_feature])

        pred = self.clf.predict(feature)
        cat = pred.tolist()[0]
        score = self.cat2score[cat] + 0.05  # 0-1 => 0.05, 1-2 => 0.15
        return score


def read_summary_questions(filepath, get_all_sentences=True):
    with open(filepath, 'r') as fin:
        dataset = Serializer.parse(fin, DataSet)

    summary_questions = []
    for question in dataset.questions:
        if question.type == "summary":
            summary_questions.append(question)

    summary_questions = vectorize.get_all_sentences(summary_questions)

    return summary_questions


def load_and_featurize(data_path, feature_type, label_type):

    summary_type_questions = read_summary_questions(data_path)
    print 'Total summary-type questions: ', len(summary_type_questions)

    # Get Feature and Label
    all_featurizers, all_features = vectorize.get_features(
        summary_type_questions, feature_type=feature_type)
    labels = vectorize.get_labels(
        summary_type_questions, label_type=label_type)

    X = hstack(all_features).toarray()
    Y = np.array(labels)
    return X, Y, all_featurizers


def train(opt):
    data_dir = opt["--data-dir"]
    train_path = os.path.join(data_dir, "summary.train.json")

    feature_type = opt["--feature"]
    label_type = opt["--label"]

    X_train, Y_train, all_featurizers = load_and_featurize(
        train_path, feature_type, label_type)

    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)

    interval = float(opt["--interval"])
    Y_train_bin, cat2score = score_to_bin(Y_train, interval)

    model_type = opt["--model"]
    if model_type == "LogisticRegression":
        clf = LogisticRegression()
    elif model_type == "LinearSVC":
        clf = LinearSVC()
    elif model_type == "MLPClassifier":
        clf = MLPClassifier()
    else:
        raise ValueError("Unknown model: {}".format(model_type))

    clf.fit(X_train, Y_train_bin)
    train_acc = evaluate(clf, X_train, Y_train_bin)

    print("train_acc", train_acc)

    # Save Model
    model = ModelWrapper(
        all_featurizers,
        clf,
        cat2score
    )

    save_dir = opt["--save-dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = "{}_{}_{}_{}.pickle".format(
        model_type, feature_type, label_type, interval)
    save_path = os.path.join(save_dir, model_name)

    with open(save_path, "wb") as fout:
        pickle.dump(model, fout)


def featurize(summary_questions, all_featurizers, label_type):
    """
    Featurize with given featurizer
    """
    question_list = []
    sentence_list = []
    for question in summary_questions:
        for sentence in question.sentences:
            question_list.append(question.body)
            sentence_list.append(sentence)

    question_features = all_featurizers[0].transform(question_list)
    sentence_features = all_featurizers[1].transform(sentence_list)

    labels = vectorize.get_labels(summary_questions, label_type=label_type)

    X = hstack([question_features, sentence_features]).toarray()
    Y = np.array(labels)
    return X, Y


def test(opt):
    """ Example Usage of Testing
    """

    data_dir = opt["--data-dir"]
    valid_path = os.path.join(data_dir, "summary.valid.json")
    test_path = os.path.join(data_dir, "summary.test.json")

    valid_questions = read_summary_questions(valid_path)
    test_questions = read_summary_questions(test_path)

    model_path = opt["--model-path"]
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    label_type = opt["--label"]
    X_valid, Y_valid = featurize(
        valid_questions, model.featurizers, label_type)
    print("X_valid", X_valid.shape, "Y_valid", Y_valid.shape)
    X_test, Y_test = featurize(
        test_questions, model.featurizers, label_type)
    print("X_test", X_test.shape, "Y_test", Y_test.shape)

    cat2score = model.cat2score
    interval = cat2score[1] - cat2score[0]

    Y_valid_bin, _ = score_to_bin(Y_valid, interval)
    Y_test_bin, _ = score_to_bin(Y_test, interval)

    clf = model.clf
    valid_acc = evaluate(clf, X_valid, Y_valid_bin)
    print('valid_acc', valid_acc)

    test_acc = evaluate(clf, X_test, Y_test_bin)
    print('test_acc', test_acc)


def example(opt):
    model_path = opt["--model-path"]
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    question = 'What is the effect of TRH on myocardial contractility?'
    sentence = 'Acute intravenous administration of TRH to rats with ischemic cardiomyopathy caused a significant increase in heart rate, mean arterial pressure, cardiac output, stroke volume, and cardiac contractility'

    print("Question:", question)
    print("Sentence:", sentence)

    score = model.score(question, sentence)
    print("Score:", score)


if __name__ == "__main__":
    opt = docopt(__doc__)

    if opt["train"]:
        train(opt)
    elif opt["test"]:
        test(opt)
    elif opt["example"]:
        example(opt)
