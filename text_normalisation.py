import pandas as pd
from data_utils import save_df_to_tsv, gen_vocab, gen_config_yaml
from pathlib import Path

# MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
manifest = {c: [] for c in MANIFEST_COLUMNS}
# rootpath = "/Users/bogumiladubel/Documents/BA/data/asr/common_voice/de/"
rootpath = "/Users/bdubel/Documents/ZHAW/BA/data/swiss_all/"
file = "train_st_ch_de.tsv"
df = pd.read_table(rootpath + "original/" + file)
train_text = []
# root_path_data = "/Users/bogumiladubel/Documents/BA/repos/st_ch_de/"
root_path_data = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/"

ALLOWED_CHARS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ä', 'ö', 'ü',
    ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9'
}


def preprocess_transcript(transcript):
    transcript = transcript.lower()
    transcript = transcript.replace('ß', 'ss')
    transcript = transcript.replace('-', ' ')
    transcript = transcript.replace('–', ' ')
    # Replace additional characters from your training set here
    # Example: transcript = transcript.replace('á', 'a')
    transcript = ''.join([char for char in transcript if char in ALLOWED_CHARS])
    return transcript.strip()


for t in range(df.shape[0]):
    if t % 1000 == 0:
        print(t, " von ", df.shape[0])
    manifest["id"].append(df.values[t][0])
    manifest["audio"].append(df.values[t][1])
    manifest["n_frames"].append(df.values[t][2])
    target = preprocess_transcript(str(df.values[t][3]))
    train_text.append(target)
    manifest["tgt_text"].append(target)
    # manifest["speaker"].append(df.values[t][4])

df = pd.DataFrame.from_dict(manifest)
save_df_to_tsv(df, Path(rootpath) / f"{file}")


def gen_voc(train_text, spm_filename_prefix):
    f = open(Path(root_path_data) / "test.txt", "a")
    for t in train_text:
        f.write(" ".join(t) + "\n")
    print(f.name)
    gen_vocab(Path(f.name), Path(root_path_data) / spm_filename_prefix)


task = "asr_de"
spm_filename_prefix = f"spm_char_{task}"
# Generate config YAML
gen_config_yaml(
    Path(root_path_data),
    spm_filename_prefix + ".model",
    yaml_filename=f"config_{task}.yaml",
    specaugment_policy="lb",
    )
# generating vocabulary
if len(train_text) > 0:
    gen_voc(train_text, spm_filename_prefix)

