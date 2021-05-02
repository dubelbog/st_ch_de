import pandas as pd
import nltk.translate.bleu_score as blue
from jiwer import wer
from fairseq_cli_stchde.helper_utils import generate_manifest

file_path = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/100/avg_evaluation.tsv"
df = pd.read_table(file_path)

path_bleu = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/blue.csv"
path_wer = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/wer.csv"
path_added = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/added.tsv"
path_missing = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/missing.tsv"

MANIFEST_COLUMNS = ["BLEU", "WER", "diff_ref_hyp", "nr_1", "diff_hyp_ref", "nr_2", "reference", "hypothese", "id"]
DOCS_COLUMNS = ['counts', 'word']
doc_df = {c: [] for c in DOCS_COLUMNS}

manifest = {c: [] for c in MANIFEST_COLUMNS}

missing_words = []
added_words = []


def get_tokens(sentence):
    tokens = []
    for word in sentence.split():
        tokens.append(word)
    return tokens


for text in range(df.shape[0]):
    manifest["id"].append(df.values[text][0])
    reference = df.values[text][1]
    manifest["reference"].append(reference)
    hypothese = df.values[text][2]
    manifest["hypothese"].append(hypothese)
    ref_tokens = get_tokens(reference)
    hyp_tokens = get_tokens(hypothese)
    print("hypothese: ", hypothese)
    print("reference: ", reference)
    bleu = blue.sentence_bleu([ref_tokens], hyp_tokens)
    print("bleu: ", bleu)
    manifest["BLEU"].append(round(bleu, 3))
    error = wer(reference, hypothese)
    manifest["WER"].append(round(error, 2))
    diff_ref_hyp = set(ref_tokens).difference(hyp_tokens)
    manifest["diff_ref_hyp"].append(" ".join(diff_ref_hyp))
    for value in list(diff_ref_hyp):
        missing_words.append(value)
    manifest["nr_1"].append(len(diff_ref_hyp))
    diff_hyp_ref = set(hyp_tokens).difference(ref_tokens)
    for value in list(diff_hyp_ref):
        added_words.append(value)
    manifest["diff_hyp_ref"].append(" ".join(diff_hyp_ref))
    manifest["nr_2"].append(len(diff_hyp_ref))


def get_words_count(words, data_doc, path):
    list_to_set = set(words)
    map_structure = []
    for word in list_to_set:
        counts = words.count(word)
        map_structure.append([counts, word])
    df_ = pd.DataFrame(map_structure, columns=['counts', 'word'])
    df_sorted = df_.sort_values(by='counts', ascending=False)
    for pos in range(df_sorted.shape[0]):
        data_doc['counts'].append(df_sorted.values[pos][0])
        data_doc['word'].append(df_sorted.values[pos][1])
    generate_manifest(data_doc, path)


# get_words_count(missing_words, doc_df, path_missing)
# get_words_count(added_words, doc_df, path_added)

generate_manifest(manifest, path_bleu)
df = pd.read_table(path_bleu)
sorted_df = df.sort_values(by='BLEU', ascending=False)
sorted_df.to_csv(path_bleu, index=False, header=True)

sorted_df = df[["WER", "BLEU", "diff_ref_hyp", "diff_hyp_ref", "reference", "hypothese", "id"]]
sorted_df = sorted_df.sort_values(by='WER', ascending=True)
sorted_df.to_csv(path_wer, index=False, header=True)
