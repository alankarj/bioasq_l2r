from rouge import Rouge as RougeLib
from nltk.tokenize import word_tokenize


class SimilarityJaccard(object):
    def __init__(self, stopWords, stemmer=None):
        self.stopWords = stopWords
        self.r = RougeLib()
        self.stemmer = stemmer

    def calculateSimilarity(self, s1, s2):
        # s2 is assumed to be a set of tokens
        set1 = set([
            i.lower() for i in word_tokenize(s1)
            if i.lower() not in self.stopWords
        ])
        set2 = s2
        return float(len(set1.intersection(set2))) / len(set1.union(set2))

    def calculateRouge(self, s1, s2):
        sent1 = s1
        sent2 = " ".join(s2)
        # print "Sentence-1", sent1
        # print "Sentence-2", sent2
        return self.r.get_scores(sent1, sent2)[0]['rouge-2']['r']

    def caculateSimilarityWithStem(self, s1, s2):
        set1 = set([self.stemmer.stem(i.lower()) for i in word_tokenize(s1) if i.lower() not in self.stopWords])
        set2 = set([self.stemmer.stem(i.lower()) for i in s2])
        return float(len(set1.intersection(set2))) / len(set1.union(set2))
