import pandas as pd
from fairseq_cli_stchde.helper_utils import generate_manifest

DOCS_COLUMNS = ['id', 'audio', 'n_frames', 'tgt_text', 'speaker']
doc_df = {c: [] for c in DOCS_COLUMNS}

file_path = "test_asr_de_v.tsv"
df = pd.read_table(file_path)

for text in range(df.shape[0]):
    doc_df["id"].append(df.values[text][0])
    audio_path = str(df.values[text][1]).replace('/Users/bdubel/Documents/ZHAW/BA/data/comon_voice_de',
                                                 '/cluster/home/dubelbog/data/asr_de')
    print(audio_path)
    print(text, " von ", df.shape[0])
    doc_df["audio"].append(audio_path)
    doc_df["n_frames"].append(df.values[text][2])
    doc_df["tgt_text"].append(df.values[text][3])
    doc_df["speaker"].append(df.values[text][4])

generate_manifest(doc_df, "test_asr_de.tsv")

