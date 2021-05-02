import nltk.translate.bleu_score as blue
from jiwer import wer
import fastwer
import os
import pandas as pd

# path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss"
# path_eval = "/resources/swiss/100"
path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/all"
# file = "evaluation125.tsv"
file = "avg_evaluation_all.tsv"
ground_truth = []
list_of_references = []
hypotheses = []

eval_file = os.path.join(path_eval, file)
df = pd.read_table(eval_file)

for text in range(df.shape[0]):
    target = df.values[text][1]
    ref = [target]
    list_of_references.append(ref)
    ground_truth.append(target)
    hypotheses.append(df.values[text][2])

score_300 = blue.corpus_bleu(list_of_references, hypotheses)
print("NLTK Blue: ", score_300)

score = blue.corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False)
print("NLTK Blue, weights changed: ", score)

error = wer(ground_truth, hypotheses)
print("WER jiwer: ", error)

error = fastwer.score(ground_truth, hypotheses)
print("WER fastwer corpus level: ", error)
error = fastwer.score(ground_truth, hypotheses, char_level=True)
print("WER fastwer char level: ", error)
