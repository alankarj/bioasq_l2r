from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import copy


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


def get_labels(summary_type_questions, label_type):
    print "Getting labels..."
    all_scores = list()
    if label_type == "JACCARD":
        stopWords = set(stopwords.words('english'))
        similarity = SimilarityJaccard(stopWords)

        for i, question in enumerate(summary_type_questions):
            print "Question-", i

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

        print "Number of scores: ", len(all_scores)
        return all_scores
    else:
        raise ValueError("Unknown label_type: {}".format(label_type))


def get_features(summary_type_questions,
                 feature_type="COUNT",
                 sentence_only=False):
    print "Getting features..."
    sentence_list = list()
    question_list = list()
    all_featurizers = list()
    all_features = list()
    for i, question in enumerate(summary_type_questions):
        print "Question-", i
        for sentence in question.sentences:
            question_list.append(question.body)
            sentence_list.append(sentence)

    if feature_type == "COUNT":
        sent_featurizer = CountVectorizer(max_features=10000)
    elif feature_type == "TF-IDF":
        sent_featurizer = TfidfVectorizer(max_features=10000)
    else:
        raise ValueError("Unknown feature_type: {}".format(feature_type))

    all_featurizers.append(sent_featurizer)

    if not sentence_only:
        question_featurizer = copy.deepcopy(sent_featurizer)
        all_featurizers.append(question_featurizer)

    sent_features = all_featurizers[0].fit_transform(sentence_list)
    all_features.append(sent_features)

    if not sentence_only:
        question_features = all_featurizers[1].fit_transform(question_list)
        all_features.append(question_features)

    return all_featurizers, all_features
