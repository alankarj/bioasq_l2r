import sys
from deiis.model import Serializer, DataSet

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python featurize.py <data.json>'
        exit(1)

    # filename = 'data/training.json'
    filename = sys.argv[1]
    print 'Processing ' + filename

    with open(filename, 'r') as fp:
        dataset = Serializer.parse(fp, DataSet)

    print dataset

    num_summary_ques = 0
    for question in dataset.questions:
        if question.type == 'summary':
            num_summary_ques += 1
            print question.id + ': ' + question.body + ': ' + question.type

            for snippet in question.snippets:
                print snippet.text

    print 'Total questions: ', len(dataset.questions)
    print 'Total summary-type questions: ', num_summary_ques
