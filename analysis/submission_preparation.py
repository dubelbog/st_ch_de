import pandas as pd
from pathlib import Path
from num2words import num2words

# root_path_resource = "/resources/swiss/"
# eval = root_path_resource + "all_dialects/"
# eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/asr_parl/best/avg_asr_parl.tsv"
eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/ensemble/tuning_parl_dial_ch.tsv"
# eval = root_path_resource + "all_dialects/small.tsv"
pred = pd.read_table(eval)

path = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/submission_13.csv"
f = open(Path(path), "w")
f.write("path" + "," + "sentence" + '\n')


def num_2_words(prediction, counter):
    for word in prediction.split(" "):
        if word.isdigit():
            print(prediction)
            counter = counter + 1
            prediction = prediction.replace(" " + word + " ", " " + num2words(word, lang='de') + " ")
            print(prediction)
    return prediction, counter


counter = 0
for text in range(pred.shape[0]):
    path = pred.values[text][0]
    # sentence, counter = num_2_words(pred.values[text][2], counter)
    f.write(path + "," + pred.values[text][2] + '\n')

print(counter)

