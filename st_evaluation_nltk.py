import nltk.translate.bleu_score as blue
from jiwer import wer
import fastwer
import os
import pandas as pd

path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss"
file = "evaluation300.tsv"
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

score = blue.corpus_bleu(list_of_references, hypotheses, weights=(1, 0.0, 0.0, 0.0), smoothing_function=None, auto_reweigh=False)
print("NLTK Blue, weigts changed: ", score)

error = wer(ground_truth, hypotheses)
print("WER jiwer: ", error)

error = fastwer.score(ground_truth, hypotheses)
print("WER fastwer corpus level: ", error)
error = fastwer.score(ground_truth, hypotheses, char_level=True)
print("WER fastwer char level: ", error)
