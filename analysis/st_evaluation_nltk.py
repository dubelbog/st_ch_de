import nltk.translate.bleu_score as blue
from jiwer import wer
import fastwer
import os
import pandas as pd
from pathlib import Path

# path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss"
# path_eval = "/resources/swiss/100"
path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/st_parl_dial"
# file = "evaluation125.tsv"
# file = "avg_evaluation_all.tsv"
nltk = open(Path(path_eval) / "overview/nltk_scores.csv", "a")
wer_f = open(Path(path_eval) / "overview/wer_scores.csv", "a")
cer_f = open(Path(path_eval) / "overview/cer_scores.csv", "a")

for file in os.listdir(path_eval):
    if file.endswith(".tsv"):
        print(file)
        print("------------")
        name = file.replace("evaluation", "").replace(".tsv", "")
        ground_truth = []
        list_of_references = []
        hypotheses = []
        hyps = []

        eval_file = os.path.join(path_eval, file)
        df = pd.read_table(eval_file)

        for text in range(df.shape[0]):
            target = df.values[text][1]
            list_of_references.append([target.split(" ")])
            pred = df.values[text][2]
            hyps.append(pred.split(" "))

            ground_truth.append(target)
            hypotheses.append(pred)

        score_300 = blue.corpus_bleu(list_of_references, hyps)
        print("NLTK Blue: ", score_300)
        nltk.write(name + ";" + str(score_300) + '\n')

        # score = blue.corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False)
        # print("NLTK Blue, weights changed: ", score)

        error = wer(ground_truth, hypotheses)
        print("WER jiwer: ", error)

        error = fastwer.score(ground_truth, hypotheses)
        print("WER fastwer corpus level: ", error)
        wer_f.write(name + ";" + str(error) + '\n')
        error = fastwer.score(ground_truth, hypotheses, char_level=True)
        print("WER fastwer char level: ", error)
        cer_f.write(name + ";" + str(error) + '\n')

nltk.close()
wer_f.close()
cer_f.close()

