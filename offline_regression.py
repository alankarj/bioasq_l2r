""" Offline Supervised Regressin Model Train and Testing
Usage:
    offline_regression.py train [options]
    offline_regression.py baseline [options]
    offline_regression.py test --model-path=<file> [options]
    offline_regression.py example --model-path=<file> [options]

Options:
    --data-dir=<file>       Directory to data [default: ./data/summary]
    --sentence-only         Sentence only features
    --question-only         Question only features 
    -f --feature=<str>      Feature type to use [default: TF-IDF]
    -l --label=<str>        Label type to use [default: JACCARD]
    -s --scale=<int>        Scale scores and cut off to integer [default: 100]
    -i --interval=<float>   Bucket interval [default: 0.1]
    --model=<str>           Model type to use [default: LinearRegression]
    --seed=<int>            Seed number for random generator [default: 11731]
    --save-dir=<file>       Directory to save trained model [default: ./save_regression]
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
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import LinearSVR, SVR
from sklearn.utils.class_weight import compute_sample_weight

from deiis.model import DataSet, Serializer


def scale_scores(Y, scale=100):
    """ Scale and quantize scores to integer
    """
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    # Quantize
    Y = Y * scale
    if scale > 1:  # If scale up, we quantize to integer
        Y = np.round(Y)
    return Y


class RegressionModelWrapper(object):
    """
    Includes
    - featurizer
    - classifier
    - label to score dict
    """

    def __init__(self, featurizers, label_type, clf, scale):
        self.featurizers = featurizers
        self.label_type = label_type
        self.clf = clf
        self.scale = scale

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

        pred_scale = self.clf.predict(feature)

        pred_score = scale_scores(pred_scale, 1./self.scale)
        return pred_score[0]


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

    print("Featurizing")
    X_train = featurize(train_questions, all_featurizers,
                        sentence_only, question_only, train=True)

    Y_train = get_labels(train_questions, label_type)
    print("X_train", X_train.shape, "Y_train", Y_train.shape)

    scale = int(opt['--scale'])

    Y_train_scale = scale_scores(Y_train, scale)
    # Load model
    model_type = opt["--model"]
    print("Model:", model_type)
    if model_type == "LinearRegression":
        clf = LinearRegression()
    elif model_type == "LinearSVR":
        clf = LinearSVR(verbose=2)
    elif model_type == "SVR":
        clf = SVR(verbose=2)
    else:
        raise ValueError("Unknown model: {}".format(model_type))

    # Train
    print("Start training")
    # Create sample weights
    interval = float(opt['--interval'])
    Y_train_bin, cat2score = score_to_bin(Y_train, interval)
    sample_weight = compute_sample_weight("balanced", Y_train_bin)

    clf.fit(X_train, Y_train_scale, sample_weight=sample_weight)

    Y_train_pred_scale = clf.predict(X_train)
    Y_train_pred = scale_scores(Y_train_pred_scale, scale=1./scale)

    print("Scaled:")

    mae = mean_absolute_error(Y_train_scale, Y_train_pred_scale)
    mse = mean_squared_error(Y_train_scale, Y_train_pred_scale)
    print("mean absolute error", mae)
    print("mean squared error", mse)

    print("Unscaled:")
    mae = mean_absolute_error(Y_train, Y_train_pred)
    mse = mean_squared_error(Y_train, Y_train_pred)
    print("mean absolute error", mae)
    print("mean squared error", mse)

    # Save Model
    obj = (all_featurizers, label_type, clf, scale)

    save_dir = opt["--save-dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if sentence_only:
        feature_type += "_s_only"
    elif question_only:
        feature_type += "_q_only"

    model_name = "{}_{}_{}_{}".format(
        model_type, feature_type, label_type, scale)

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
        (all_featurizers, label_type, clf, scale) = pickle.load(fin)

    question_only = bool(opt["--question-only"])
    sentence_only = bool(opt["--sentence-only"])

    X_valid = featurize(valid_questions, all_featurizers,
                        sentence_only, question_only)
    Y_valid = get_labels(valid_questions, label_type)

    print("X_valid", X_valid.shape, "Y_valid", Y_valid.shape)

    Y_valid_scale = scale_scores(Y_valid, scale)

    Y_valid_pred_scale = clf.predict(X_valid)

    Y_valid_pred_scale = clf.predict(X_valid)
    Y_valid_pred = scale_scores(Y_valid_pred_scale, scale=1./scale)

    print("Scaled:")

    mae = mean_absolute_error(Y_valid_scale, Y_valid_pred_scale)
    mse = mean_squared_error(Y_valid_scale, Y_valid_pred_scale)
    print("mean absolute error", mae)
    print("mean squared error", mse)

    print("Unscaled:")
    mae = mean_absolute_error(Y_valid, Y_valid_pred)
    mse = mean_squared_error(Y_valid, Y_valid_pred)
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
