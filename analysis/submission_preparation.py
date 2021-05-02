import pandas as pd
from pathlib import Path

# root_path_resource = "/resources/swiss/"
# eval = root_path_resource + "all_dialects/"
eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/asr_parl/best/avg_asr_parl.tsv"
# eval = root_path_resource + "all_dialects/small.tsv"
pred = pd.read_table(eval)

path = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/submission_asr_parl.csv"
f = open(Path(path), "w")
f.write("path" + "," + "sentence"+ '\n')

for text in range(pred.shape[0]):
    path = pred.values[text][0]
    sentence = pred.values[text][2]
    f.write(path + "," + sentence + '\n')

