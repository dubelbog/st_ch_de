import pandas as pd
from pathlib import Path

file_path = "/Users/bdubel/Documents/ZHAW/BA/data/comon_voice_de/dev_small.tsv"
pred = pd.read_table(file_path)

path = "/Users/bdubel/Documents/ZHAW/BA/data/comon_voice_de/dev_small_test.tsv"
f = open(Path(path), "w")
f.write("sentence" + '\n')

for text in range(pred.shape[0]):
    sentence = pred.values[text][2]
    f.write(sentence + '\n')
