"""
Split ./data/training.json to train & valid
"""
import json
import os
import random


if __name__ == "__main__":
    filepath = "./data/training.json"

    save_dir = "./data"

    with open(filepath, 'r') as fin:
        obj = json.loads(fin.read())
        questions = obj["questions"]

    # Filter summaries
    summary_questions = [q for q in questions if q['type'] == "summary" or q['type'] == "factoid"]

    # assert len(summary_questions) == 400

    random.seed(11731)
    random.shuffle(summary_questions)

    train_questions = summary_questions[:664]
    valid_questions = summary_questions[664:]

    qs = [train_questions, valid_questions]
    splits = ["train", "valid"]
    for split, q in zip(splits, qs):
        print(split, len(q))
        split_path = os.path.join(save_dir, "summary_factoid.{}.json".format(split))
        print("Saving to {}".format(split_path))
        with open(split_path, 'w') as fout:
            fout.write(json.dumps({"questions": q}))
