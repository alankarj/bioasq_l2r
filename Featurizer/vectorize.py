from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pythonrouge.pythonrouge import Pythonrouge


class SimilarityJaccard(object):
    def __init__(self, stopWords):
        self.stopWords = stopWords

    def calculateSimilarity(self, s1, s2):
        # s2 is assumed to be a set of tokens
        set1 = set([i.lower() for i in word_tokenize(s1) if i.lower() not in self.stopWords])
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
    all_scores = list()
    if label_type == "JACCARD":
        stopWords = set(stopwords.words('english'))
        similarity = SimilarityJaccard(stopWords)

        for i, question in enumerate(summary_type_questions):
            print "Question-", i

            list_of_sets = []

            if type(question.ideal_answer) == list:
                for ideal_answer in question.ideal_answer:
                    list_of_sets.append(set([i.lower() for i in word_tokenize(ideal_answer) if i.lower() not in stopWords]))
            else:
                list_of_sets.append(set([i.lower() for i in word_tokenize(question.ideal_answer) if i.lower() not in stopWords]))

            for sentence in question.sentences:
                scores = []
                for s2 in list_of_sets:
                    scores.append(similarity.calculateSimilarity(sentence, s2))

                all_scores.append(sum(scores)/len(scores))

    else:
        for i, question in enumerate(summary_type_questions):
            print "Question-", i

            list_of_sets = []

            if type(question.ideal_answer) == list:
                for ideal_answer in question.ideal_answer:
                    list_of_sets.append(set([i.lower() for i in word_tokenize(ideal_answer) if i.lower() not in stopWords]))
            else:
                list_of_sets.append(set([i.lower() for i in word_tokenize(question.ideal_answer) if i.lower() not in stopWords]))

            for sentence in question.sentences:
                scores = []
                for s2 in list_of_sets:
                    scores.append(similarity.calculateSimilarity(sentence, s2))

                all_scores.append(sum(scores)/len(scores))

        print "Number of scores: ", len(all_scores)
