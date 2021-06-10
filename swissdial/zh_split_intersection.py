import pandas as pd
from data_utils import save_df_to_tsv
from pathlib import Path

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
manifest = {c: [] for c in MANIFEST_COLUMNS}
rootpath = "/Users/bogumiladubel/Documents/BA/data/st/eth_swiss_dialects/split_dialect/swissdial/"

df_zh = pd.read_table(rootpath + "test_zh.tsv")
df_vs = pd.read_table(rootpath + "test_vs.tsv")

intersection_zh = pd.merge(df_zh, df_vs, on=['tgt_text'], how='inner')

intersection_zh = intersection_zh[['id_x', 'audio_x', 'n_frames_x', 'tgt_text']]
intersection_zh = intersection_zh.rename(columns={"id_x": "id", "audio_x": "audio", "n_frames_x": "n_frames"})
intersection_zh = intersection_zh.drop_duplicates()


def write_manifest(frame, name):
    length = frame.shape[0]
    for pos in range(length):
        print(pos, " von ", length)
        manifest["id"].append(frame.values[pos][0])
        manifest["audio"].append(frame.values[pos][1])
        manifest["n_frames"].append(frame.values[pos][2])
        manifest["tgt_text"].append(frame.values[pos][3])
    df_manifest = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df_manifest, Path(rootpath) / f"{name}.tsv")


write_manifest(intersection_zh, "test_intersection_zh")


