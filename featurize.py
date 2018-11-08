import sys
from deiis.model import Serializer, DataSet
from Featurizer import vectorize

LABEL_TYPE = "JACCARD"  # Use "ROUGE" for ROUGE-2
FEATURE_TYPE = "TF-IDF"  # Use "TF-IDF" for TF-IDF

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python featurize.py <data.json>'
        exit(1)

    # filename = 'data/training.json'
    filename = sys.argv[1]
    print 'Processing ' + filename

    with open(filename, 'r') as fp:
        dataset = Serializer.parse(fp, DataSet)

    summary_type_questions = []
    for question in dataset.questions:
        if question.type == 'summary':
            summary_type_questions.append(question)
            print question.id + ': ' + question.body + ': ' + question.type

    print 'Total questions: ', len(dataset.questions)
    print 'Total summary-type questions: ', len(summary_type_questions)

    summary_type_questions = vectorize.get_all_sentences(summary_type_questions)
    all_featurizers, all_features = vectorize.get_features(summary_type_questions, feature_type=FEATURE_TYPE)
    labels = vectorize.get_labels(summary_type_questions, label_type=LABEL_TYPE)