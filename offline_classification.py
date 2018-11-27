""" Offline Supervised Model Train and Testing
Usage:
    offline_classification.py train [options]
    offline_classification.py baseline [options]
    offline_classification.py test --model-path=<file> [options]
    offline_classification.py example --model-path=<file> [options]
    offline_classification.py visualize [options]

Options:
    --data-dir=<file>       Directory to data [default: ./data/summary]
    --sentence-only         Sentence only features
    --question-only         Question only features 
    -f --feature=<str>      Feature type to use [default: TF-IDF]
    -l --label=<str>        Label type to use [default: JACCARD]
    -i --interval=<float>   Interval for binary classification [default: 0.1]
    --model=<str>           Model type to use [default: LinearSVC]
    --seed=<int>            Seed number for random generator [default: 11731]
    --save-dir=<file>       Directory to save trained model [default: ./save_classification]
    --model-path=<file>     Path to model pickle [default: ./save/LinearSVC_TF-IDF_JACCARD_0.1.pickle]
"""
import copy
import json
import math
import os
import pickle
import random

import numpy as np
from docopt import docopt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from deiis.model import DataSet, Serializer


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


class ModelWrapper(object):
    """
    Includes
    - featurizer
    - classifier
    - label to score dict
    """

    def __init__(self, featurizers, label_type, clf, cat2score):
        self.featurizers = featurizers
        self.label_type = label_type
        self.clf = clf
        self.cat2score = cat2score

    def score(self, question, sentence):

        # Featurizer
        if len(self.featurizers) == 3:
            sentence_featurizer, question_featurizer, pca = self.featurizers
        else:
            sentence_featurizer, question_featurizer = self.featurizers

        sentence_feature = sentence_featurizer.transform([sentence])
        question_feature = question_featurizer.transform([question])

        feature = hstack([sentence_feature, question_feature]).toarray()

        if len(self.featurizers) == 3:
            feature = pca.transform(feature)

        pred = self.clf.predict(feature)
        cat = pred.tolist()[0]

        interval = self.cat2score[1] - self.cat2score[0]

        score = self.cat2score[cat] + \
            float(interval)/2  # 0-1 => 0.05, 1-2 => 0.15
        return score


def save_to_pickle(obj, save_name):
    """
    Save Model
    """
    with open(save_name, 'wb') as fout:
        pickle.dump(obj, fout)


def get_sentences(question):
    sentences = []
    for snippet in question.snippets:
        text = unicode(snippet.text).encode("ascii", "ignore")
        if text == "":
            continue
        try:
            sentences += sent_tokenize(text)
        except:
            sentences += text.split(". ")  # Notice the space after the dot
    return sentences


def get_all_sentences(summary_type_questions):
    #
    num_sentences = 0
    for question in summary_type_questions:
        question.sentences = get_sentences(question)

        # print "##############################################################"
        # print question.ideal_answer

        # for sentence in question.sentences:
        #     print similarity.calculateSimilarity(sentence, question.ideal_answer[0])

        num_sentences += len(question.sentences)

    print 'Total number of sentences: ', num_sentences
    return summary_type_questions


def read_summary_questions(filepath):
    with open(filepath, 'r') as fin:
        dataset = Serializer.parse(fin, DataSet)

    summary_questions = []
    for question in dataset.questions:
        if question.type == "summary":
            summary_questions.append(question)

    summary_questions = get_all_sentences(summary_questions)
    return summary_questions


def create_featurizers(feature_type):
    """Create featurizers and return as list 
        1. sentence featurizer
        2. question featurizer
        3. pca 
    """
    # Sentence & Question
    if feature_type == "COUNT":
        sent_featurizer = CountVectorizer(max_features=10000)
    elif feature_type == "TF-IDF":
        sent_featurizer = TfidfVectorizer(max_features=10000)
    else:
        raise ValueError("Unknown feature_type: {}".format(feature_type))

    question_featurizer = copy.deepcopy(sent_featurizer)

    # PCA
    pca = PCA(n_components=300)
    all_featurizers = [sent_featurizer, question_featurizer, pca]
    return all_featurizers


class SimilarityJaccard(object):
    def __init__(self, stopWords):
        self.stopWords = stopWords

    def calculateSimilarity(self, s1, s2):
        # s2 is assumed to be a set of tokens
        set1 = set([
            i.lower() for i in word_tokenize(s1)
            if i.lower() not in self.stopWords
        ])
        set2 = s2
        return float(len(set1.intersection(set2))) / len(set1.union(set2))


def get_labels(summary_type_questions, label_type):
    print "Getting labels..."
    all_scores = list()
    if label_type == "JACCARD":
        stopWords = set(stopwords.words('english'))
        similarity = SimilarityJaccard(stopWords)

        for i, question in enumerate(summary_type_questions):
            #print "Question-", i

            list_of_sets = []

            if type(question.ideal_answer) == list:
                for ideal_answer in question.ideal_answer:
                    list_of_sets.append(
                        set([
                            i.lower() for i in word_tokenize(ideal_answer)
                            if i.lower() not in stopWords
                        ]))
            else:
                list_of_sets.append(
                    set([
                        i.lower() for i in word_tokenize(question.ideal_answer)
                        if i.lower() not in stopWords
                    ]))

            for sentence in question.sentences:
                scores = []
                for s2 in list_of_sets:
                    scores.append(similarity.calculateSimilarity(sentence, s2))

                all_scores.append(sum(scores) / len(scores))
    else:
        raise ValueError("Unknown label type: {}".format(label_type))

    all_scores = np.array(all_scores)
    return all_scores


def featurize(summary_questions, all_featurizers, sentence_only=False, question_only=False, train=False):
    """
    Featurize with given featurizer
    """
    print("[featurize]", "train", train)
    # Process into question + sentence data samples
    question_list = []
    sentence_list = []
    for question in summary_questions:
        for sentence in question.sentences:
            question_list.append(question.body)
            sentence_list.append(sentence)

    # Process word tokens into feature array
    sent_featurizer, question_featurizer, pca = all_featurizers

    if train:
        sent_featurizer.fit(sentence_list)
        question_featurizer.fit(question_list)

    sentence_features = all_featurizers[0].transform(sentence_list)
    question_features = all_featurizers[1].transform(question_list)

    if sentence_only:
        X = sentence_features.toarray()
    elif question_only:
        X = question_features.toarray()
    else:
        X = hstack([sentence_features, question_features]).toarray()

    # PCA the feature array
    if train:
        pca.fit(X)

    X = pca.transform(X)
    return X


def convert_cat2score(Y_pred, cat2score):
    if len(cat2score) == 1:
        raise Warning("There is only 1 category")
    interval = cat2score[1] - cat2score[0]
    scores = []
    for y in Y_pred:
        cat = cat2score[y]
        s = interval * (cat + 0.5)
        scores.append(s)
    scores = np.array(scores)
    return scores


def train(opt):

    # Process data
    data_dir = opt["--data-dir"]
    train_path = os.path.join(data_dir, "summary.train.json")

    feature_type = opt["--feature"]
    label_type = opt["--label"]

    question_only = bool(opt["--question-only"])
    sentence_only = bool(opt["--sentence-only"])

    train_questions = read_summary_questions(train_path)
    all_featurizers = create_featurizers(feature_type)

    X_train = featurize(train_questions, all_featurizers,
                        sentence_only, question_only, train=True)
    Y_train = get_labels(train_questions, label_type)

    print("X_train", X_train.shape, "Y_train", Y_train.shape)

    interval = float(opt["--interval"])
    Y_train_bin, cat2score = score_to_bin(Y_train, interval)

    print("Counting labels...")
    unique, counts = np.unique(Y_train_bin, return_counts=True)
    print(dict(zip(unique, counts)))

    # Load model
    model_type = opt["--model"]
    print("Model:", model_type)
    if model_type == "LogisticRegression":
        clf = LogisticRegression()
    elif model_type == "LinearSVC":
        clf = LinearSVC()
    elif model_type == "SVC":
        clf = SVC(max_iter=-1, verbose=True)
    elif model_type == "MLPClassifier":
        clf = MLPClassifier()
    else:
        raise ValueError("Unknown model: {}".format(model_type))

    # Train
    print("Start training")
    clf.fit(X_train, Y_train_bin)

    Y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(Y_train_bin, Y_train_pred)

    print("train_acc", train_acc)

    Y_train_pred_scores = convert_cat2score(Y_train_pred, cat2score)

    mae = mean_absolute_error(Y_train, Y_train_pred_scores)
    mse = mean_squared_error(Y_train, Y_train_pred_scores)
    print("mean absolute error", mae)
    print("mean squared error", mse)

    # Save Model
    obj = (all_featurizers, label_type, clf, cat2score)

    save_dir = opt["--save-dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if sentence_only:
        feature_type += "_s_only"
    elif question_only:
        feature_type += "_q_only"

    model_name = "{}_{}_{}_{}".format(
        model_type, feature_type, label_type, interval)

    save_path = os.path.join(save_dir, model_name + ".pickle")
    print("saving model to {}".format(save_path))
    with open(save_path, "wb") as fout:
        pickle.dump(obj, fout)


def test(opt):
    """ Example Usage of Testing
    """
    data_dir = opt["--data-dir"]
    valid_path = os.path.join(data_dir, "summary.valid.json")
    valid_questions = read_summary_questions(valid_path)

    model_path = opt["--model-path"]
    with open(model_path, 'rb') as fin:
        (all_featurizers, label_type, clf, cat2score) = pickle.load(fin)

    label_type = label_type

    question_only = bool(opt["--question-only"])
    sentence_only = bool(opt["--sentence-only"])

    X_valid = featurize(valid_questions, all_featurizers,
                        sentence_only, question_only)
    Y_valid = get_labels(valid_questions, label_type)

    print("X_valid", X_valid.shape, "Y_valid", Y_valid.shape)
    interval = cat2score[1] - cat2score[0]

    Y_valid_bin, _ = score_to_bin(Y_valid, interval)

    print("Counting labels...")
    unique, counts = np.unique(Y_valid_bin, return_counts=True)
    print(dict(zip(unique, counts)))

    Y_valid_pred = clf.predict(X_valid)
    valid_acc = accuracy_score(Y_valid_bin, Y_valid_pred)

    print("Counting validation prediction...")
    unique, counts = np.unique(Y_valid_pred, return_counts=True)
    print(dict(zip(unique, counts)))

    print('valid_acc', valid_acc)

    # Mean Squared Error
    Y_valid_pred_scores = convert_cat2score(Y_valid_pred, cat2score)

    mae = mean_absolute_error(Y_valid, Y_valid_pred_scores)
    mse = mean_squared_error(Y_valid, Y_valid_pred_scores)
    print("mean absolute error", mae)
    print("mean squared error", mse)


def baseline(opt):
    """ Baseline with majority class prediction
    """
    data_dir = opt["--data-dir"]
    train_path = os.path.join(data_dir, "summary.train.json")
    valid_path = os.path.join(data_dir, "summary.valid.json")
    train_questions = read_summary_questions(train_path)
    valid_questions = read_summary_questions(valid_path)

    label_type = opt["--label"]

    Y_train = get_labels(train_questions, label_type)
    Y_valid = get_labels(valid_questions, label_type)

    interval = float(opt["--interval"])

    Y_train_bin, cat2score = score_to_bin(Y_train, interval)
    Y_valid_bin, cat2score = score_to_bin(Y_valid, interval)

    print("Train Set")
    print("Counting labels...")
    unique, counts = np.unique(Y_train_bin, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)

    majority_class = max(count_dict, key=count_dict.get)

    Y_train_pred = [majority_class] * int(Y_train_bin.size)

    train_acc = accuracy_score(Y_train_bin, Y_train_pred)

    print('train_acc', train_acc)

    # Mean Squared Error
    Y_train_pred_scores = convert_cat2score(Y_train_pred, cat2score)

    mae = mean_absolute_error(Y_train, Y_train_pred_scores)
    mse = mean_squared_error(Y_train, Y_train_pred_scores)
    print("mean absolute error", mae)
    print("mean squared error", mse)

    print("Validation Set")
    print("Counting labels...")
    unique, counts = np.unique(Y_valid_bin, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)

    majority_class = max(count_dict, key=count_dict.get)

    Y_valid_pred = [majority_class] * int(Y_valid_bin.size)

    valid_acc = accuracy_score(Y_valid_bin, Y_valid_pred)

    print('valid_acc', valid_acc)

    # Mean Squared Error
    Y_valid_pred_scores = convert_cat2score(Y_valid_pred, cat2score)

    mae = mean_absolute_error(Y_valid, Y_valid_pred_scores)
    mse = mean_squared_error(Y_valid, Y_valid_pred_scores)
    print("mean absolute error", mae)
    print("mean squared error", mse)


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


def visualize_features(opt):
    """Visualize features by TSNE
    """
    data_dir = opt["--data-dir"]
    train_path = os.path.join(data_dir, "summary.train.json")

    print("Read train")
    feature_type = opt["--feature"]
    label_type = opt["--label"]
    sentence_only = bool(opt["--sentence-only"])
    question_only = bool(opt["--question-only"])

    train_questions = read_summary_questions(train_path)
    featurizers = create_featurizers(feature_type)
    X_train = featurize(train_questions, featurizers,
                        sentence_only, question_only, train=True)
    Y_train = get_labels(train_questions, label_type)

    interval = float(opt["--interval"])
    Y_train_bin, cat2score = score_to_bin(Y_train, interval)

    print("Perform TSNE train")
    X_train_tsne = TSNE(verbose=2).fit_transform(X_train)

    print("Visualize train dataset")
    import matplotlib
    import matplotlib.pyplot as plt

    num_labels = len(cat2score)

    def plot_and_save(X, Y, num_labels, save_path):
        plt.scatter(X[:, 0], X[:, 1], c=Y,
                    cmap=plt.cm.get_cmap("jet", num_labels))
        plt.colorbar(ticks=range(num_labels))
        plt.clim(-0.5, num_labels-0.5)
        plt.savefig(save_path)

    plot_and_save(X_train_tsne, Y_train_bin, num_labels, "train.png")


if __name__ == "__main__":
    opt = docopt(__doc__)

    if opt["train"]:
        train(opt)
    elif opt["test"]:
        test(opt)
    elif opt["example"]:
        example(opt)
    elif opt["visualize"]:
        visualize_features(opt)
    elif opt["baseline"]:
        baseline(opt)
