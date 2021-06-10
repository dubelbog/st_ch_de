import pandas as pd
from data_utils import save_df_to_tsv
from pathlib import Path

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
manifest_dev = {c: [] for c in MANIFEST_COLUMNS}
manifest_test = {c: [] for c in MANIFEST_COLUMNS}
manifest_train = {c: [] for c in MANIFEST_COLUMNS}
rootpath = "/Users/bogumiladubel/Documents/BA/data/st/eth_swiss_dialects/"
file = "swissdial_all.tsv"
dev = "dev_swissdial.tsv"
test = "test_swissdial.tsv"
train = "train_swissdial.tsv"
df = pd.read_table(rootpath + "old/" + file)


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


def write_manifest(manifest, id, audio, n_frames, tgt_text):
    manifest["id"].append(id)
    manifest["audio"].append(audio)
    manifest["n_frames"].append(n_frames)
    manifest["tgt_text"].append(tgt_text)


def save_manifest(file, manifest):
    df_man = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df_man, Path(rootpath) / f"{file}.tsv")


for text in range(df.shape[0]):
    print(text, " von ", df.shape[0])
    id = str(df.values[text][0])
    nr = id.split("_")[2].replace(".wav", "")
    audio = str(df.values[text][1]).\
        replace("/cluster/home/dubelbog/data/eth_dialects", "/home/ubuntu/data/st/swissdial")
    target = preprocess_transcript(df.values[text][3])
    if 150 <= int(nr) <= 300:
        write_manifest(manifest_test, id, audio, df.values[text][2], target)
    elif 700 <= int(nr) <= 950:
        write_manifest(manifest_dev, id, audio, df.values[text][2], target)
    else:
        write_manifest(manifest_train, id, audio, df.values[text][2], target)


save_manifest(dev, manifest_dev)
save_manifest(test, manifest_test)
save_manifest(train, manifest_train)
