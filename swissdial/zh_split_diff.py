import pandas as pd
from data_utils import save_df_to_tsv
from pathlib import Path


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
manifest = {c: [] for c in MANIFEST_COLUMNS}
rootpath = "/Users/bogumiladubel/Documents/BA/data/st/eth_swiss_dialects/split_dialect/swissdial/"

frame = pd.read_csv(rootpath + "test_diff.csv")

length = frame.shape[0]
for pos in range(length):
    print(pos, " von ", length)
    manifest["id"].append(frame.values[pos][1])
    manifest["audio"].append(frame.values[pos][2])
    manifest["n_frames"].append(frame.values[pos][3])
    manifest["tgt_text"].append(frame.values[pos][4])
df_manifest = pd.DataFrame.from_dict(manifest)
name = "test_diff_zh"
save_df_to_tsv(df_manifest, Path(rootpath) / f"{name}.tsv")


