import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

refs = []
hyps = []

refsx = []
hypsx = []

# df = pd.read_table("/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/evaluation300.tsv")
# df = pd.read_table("/resources/swiss/100/avg_evaluation.tsv")
df = pd.read_table("/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/all/avg_evaluation_all.tsv")
# df = pd.read_table("/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/references/reference_prediction_tuples_with_metrics.tsv")

references = []
for text in range(df.shape[0]):
    target = df.values[text][1]
    refs.append([target.split(" ")])
    pred = df.values[text][2]
    hyps.append(pred.split(" "))

    refsx.append([target])
    hypsx.append(pred)

print(corpus_bleu(refs, hyps)) # "real" BLEU score