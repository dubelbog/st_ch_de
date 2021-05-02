import pandas as pd
from pathlib import Path

tokenizer_train = []
tokenizer_dev = []
tokenizer_predictions = []


root_path_data = "/Users/bdubel/Documents/ZHAW/BA/data/swiss_09/"
train_file = root_path_data + "train_st_ch_de.tsv"
dev_file = root_path_data + "dev_st_ch_de.tsv"
test_file = root_path_data + "test_st_ch_de_small.tsv"
root_path_resource = "/resources/swiss/"
predictions_file = root_path_resource + "evaluation300.tsv"
f = open(Path(root_path_resource) / "overview/tokens_comparison.txt", "r")

train = pd.read_table(train_file)
dev = pd.read_table(dev_file)
predictions = pd.read_table(predictions_file)


def get_tokens(df, pos):
    tokens = []
    counter = 0
    for text in range(df.shape[0]):
        target = df.values[text][pos]
        for token in target.split():
            tokens.append(token)
        tokens = list(dict.fromkeys(tokens))
        counter = counter + 1
        print("Token " + str(counter) + " von " + str(df.shape[0]))
    return sorted(tokens)


def get_diff_pred_train():
    print("Train tokenization")
    train_dict = get_tokens(train, 3)
    print("train_dict length: ", len(train_dict))

    print("Dev tokenization")
    dev_dict = get_tokens(dev, 3)
    print("dev_dict length: ", len(dev_dict))

    print("Predictions tokenization")
    predictions_dict = get_tokens(predictions, 2)
    print("predictions_dict length: ", len(predictions_dict))

    union = set(train_dict).union(dev_dict)
    print("union length: ", len(union))

    diff = set(predictions_dict).difference(union)
    print("diff length: ", len(diff))

    f.write("Difference prediction - training & dev")
    diff = sorted(diff)
    for d in diff:
        f.write(d + '\n')

    return diff


def get_diff_test_set():
    pred_test_file = root_path_resource + "all_dialects/avg_evaluation.tsv"
    pred = pd.read_table(pred_test_file)
    print("Predictions tokenization")
    pred_dict = get_tokens(pred, 2)
    print("predictions_dict length: ", len(pred_dict))

    print("Train tokenization")
    train_dict = get_tokens(train, 3)
    print("train_dict length: ", len(train_dict))

    print("Dev tokenization")
    dev_dict = get_tokens(dev, 3)
    print("dev_dict length: ", len(dev_dict))

    union = set(train_dict).union(dev_dict)
    print("union length: ", len(union))

    diff = set(pred_dict).difference(union)
    print("diff length: ", len(diff))

    file_diff = open(Path(root_path_resource) / "overview/diff_pred_minus_train.txt", "w")
    file_diff.write("Test Set all dialects: Difference prediction - training & dev")
    diff = sorted(diff)
    for d in diff:
        file_diff.write(d + '\n')


def get_diff_test_pred():
    print("Predictions tokenization")
    predictions_dict = get_tokens(predictions, 2)
    print("predictions_dict length: ", len(predictions_dict))

    print("Test tokenization")
    test_dict = get_tokens(predictions, 1)
    print("predictions_dict length: ", len(test_dict))

    diff = set(predictions_dict).difference(test_dict)
    print("diff length: ", len(diff))
    file_difference = open(Path(root_path_resource) / "overview/tokens_comparison_pred_diff_test.txt", "w")
    file_difference.write("Difference: prediction - test")
    for d in sorted(diff):
        file_difference.write(d + '\n')

    file = open(Path(root_path_resource) / "overview/tokens_comparison.txt", "r")
    diff_pred_train = []

    for line in file:
        diff_pred_train.append(line.removesuffix('\n'))

    file_intersection = open(Path(root_path_resource) / "overview/tokens_comparison_intersection_diff.txt", "w")
    file_intersection.write("Intersection (prediction - training & dev) && (prediction - test)")

    inter = set(diff).intersection(diff_pred_train)
    for d in sorted(inter):
        file_intersection.write(d + '\n')

    file = open(Path(root_path_resource) / "overview/tokens_comparison_diffTest_minus_diffTrain.txt", "w")
    file.write("Difference: (prediction - test) - (prediction - training & dev)\n")

    difference = set(diff).difference(diff_pred_train)
    for d in sorted(difference):
        file.write(d + '\n')

    file = open(Path(root_path_resource) / "overview/tokens_comparison_diffTrain_minus_diffTest.txt", "w")
    file.write("Difference: (prediction - training & dev) - (prediction - test)\n")

    difference = set(diff_pred_train).difference(diff)
    for d in sorted(difference):
        file.write(d + '\n')


get_diff_test_set()











