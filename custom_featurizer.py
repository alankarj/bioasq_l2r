import numpy as np
from nltk.corpus import stopwords
from similarity_scoring import SimilarityJaccard
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)


class CustomFeaturizer:
    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.similarity = SimilarityJaccard(self.stopWords, stemmer=None)

    def fit(self, question_list, sentence_list):
        pass

    def transform(self, question_list, sentence_list):
        feat_1 = np.zeros((len(question_list), 1))
        for i, question in enumerate(question_list):
            if question.lower().lstrip().startswith("what is"):
                feat_1[i] = 1

        print "Number of what is...questions", np.sum(feat_1, axis=0)

        feat_2 = np.zeros((len(question_list), 1))

        for i, question in enumerate(question_list):
            feat_2[i] = self.similarity.calculateSimilarity(question, word_tokenize(sentence_list[i]))

        # print "Average: ", np.mean(feat_2)
        # print "Std. Dev.: ", np.std(feat_2)
        # print "Min.: ", np.min(feat_2)
        # print "Max.: ", np.max(feat_2)
        # plot_dist(feat_2[:, 0], "similarity_q_s.pdf")

        return np.concatenate([feat_1, feat_2], axis=1)


def plot_dist(y, file_name):
    sns.distplot(y, norm_hist=True)
    plt.xlabel("Label values")
    plt.ylabel("Histogram/Density")
    plt.savefig(file_name, bbox_inches="tight")
    plt.show()
