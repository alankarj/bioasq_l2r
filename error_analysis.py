import os
import json
from deiis.model import Serializer, DataSet
import numpy as np
import csv

DATA_PATH = "./data"
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.75


def read_results(baseline_file_name, system_file_name):
    with open(os.path.join(DATA_PATH, baseline_file_name), "rb") as f:
        baseline_results = json.load(f)
    with open(os.path.join(DATA_PATH, system_file_name), "rb") as f:
        system_results = json.load(f)
    return baseline_results, system_results


def read_original_data(data_file_name):
    with open(os.path.join(DATA_PATH, data_file_name), "rb") as fp:
        dataset = Serializer.parse(fp, DataSet)

    summary_type_questions = []
    for question in dataset.questions:
        print "Question ID: " + question.id
        print "Type: " + question.type
        print "Body: " + question.body
        print "Ideal answer: " + str(question.ideal_answer)
        print "Exact answer: " + str(question.exact_answer)

        if question.type == "summary":
            summary_type_questions.append(question)
            # print question.id + ': ' + question.body + ': ' + question.type

    print 'Total questions: ', len(dataset.questions)
    print 'Total summary-type questions: ', len(summary_type_questions)

    return dataset


def print_basic_stats(dataset, rouge):
    print "Baseline average: ", np.mean(rouge["baseline"])
    print "System average: ", np.mean(rouge["system"])

    rouge_summary = {"baseline": list(), "system": list()}
    rouge_non_summary = {"baseline": list(), "system": list()}

    for i, question in enumerate(dataset.questions):
        if question.type == "summary":
            for key in rouge:
                rouge_summary[key].append(rouge[key][i])
        else:
            for key in rouge:
                rouge_non_summary[key].append(rouge[key][i])

    for key in rouge:
        print("###############################################################")
        print("Summary results: ")
        print(key + ": (%.4f, %.4f)" % (np.mean(rouge_summary[key]), np.std(rouge_summary[key])))

        print("###############################################################")
        print("Non-summary results: ")
        print(key + ": (%.4f, %.4f)" % (np.mean(rouge_non_summary[key]), np.std(rouge_non_summary[key])))

    assert len(rouge_summary["baseline"]) == len(rouge_summary["system"]) == 32
    assert len(rouge_non_summary["baseline"]) == len(rouge_non_summary["system"]) == 67


if __name__ == "__main__":
    baseline_file_name = "baseline_results"
    system_file_name = "l2r_results"
    data_file_name = "phaseB_4b_04_with_answers.json"

    baseline_results, system_results = read_results(baseline_file_name, system_file_name)
    dataset = read_original_data(data_file_name)

    results_data_map = dict()

    for i, ideal_ans in enumerate(baseline_results["ideal"]):
        results_data_map[i] = None
        for j, question in enumerate(dataset.questions):
            if ideal_ans == question.ideal_answer[0]:
                results_data_map[i] = j
                break

    print results_data_map

    rouge = dict()
    rouge["baseline"] = [br["r"] for br in baseline_results["individual"]]
    rouge["system"] = [sr["r"] for sr in system_results["individual"]]

    # Step-1: Get average + std.dev. on rouge scores for summary and non-summary questions
    print_basic_stats(dataset, rouge)

    # Step-2: Prepare a file containing (question id, question, ideal answer, generated answer (baseline)
    # , generated answer (system), rouge (baseline), rouge (system))
    with open(os.path.join(DATA_PATH, "error_analysis.tsv"), "w") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        write_str = ["Question ID", "Question Type (Fine)", "Question Type (Coarse)", "Question Body"
                     "Ideal Answer", "Baseline Answer", "L2R Answer", "ROUGE (Baseline)", "ROUGE (L2R)",
                     "Example Type (Hard/Easy/Progression/Regression)"]
        csv_writer.writerow(write_str)
        for i, ideal_ans in enumerate(baseline_results["ideal"]):
            question = dataset.questions[results_data_map[i]]
            if question.type == "summary":
                coarse_type = "summary"
            else:
                coarse_type = "non-summary"

            if rouge["baseline"][i] < MIN_THRESHOLD and rouge["system"][i] < MIN_THRESHOLD:
                example_type = "Hard"
            elif rouge["baseline"][i] >= MAX_THRESHOLD and rouge["system"][i] >= MAX_THRESHOLD:
                example_type = "Easy"
            else:
                if rouge["system"][i] >= rouge["baseline"][i]:
                    example_type = "Progression"
                else:
                    example_type = "Regression"

            write_str = [question.id,
                         question.type,
                         coarse_type,
                         question.body.encode('utf8').replace(",", ""),
                         ideal_ans.encode('utf8').replace(",", ""),
                         baseline_results["predict"][i].encode('utf8').replace(",", ""),
                         system_results["predict"][i].encode('utf8').replace(",", ""),
                         rouge["baseline"][i],
                         rouge["system"][i],
                         example_type]
            csv_writer.writerow(write_str)
