# coding=utf-8
from pydub import AudioSegment
import torchaudio
from data_utils import extract_fbank_features, save_df_to_tsv, gen_config_yaml, gen_voc
from pathlib import Path
import pandas as pd
import os
import shutil

root = "/Users/bdubel/Documents/ZHAW/BA/data/eth_ch_dialects"
mp3_root = "/Users/bdubel/Documents/ZHAW/BA/data/eth_ch_dialects/mp3"
json = "sentences_ch_de_transcribed.json"
data = pd.read_json(Path(root) / json)
df = pd.DataFrame(data)
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
test = {c: [] for c in MANIFEST_COLUMNS}
train = {c: [] for c in MANIFEST_COLUMNS}
dev = {c: [] for c in MANIFEST_COLUMNS}
ms = 1000
os.mkdir(root + "/mp3")
replace_signs = ['-', '–', "»", "«", ".", ",", "(", ")", "?", "!", "/", ":", ";", "]", "["]
split = 0.1
train_text = []


def text_processing(tgt_text):
    tgt_text = tgt_text.replace('ß', 'ss').lower()
    for char in replace_signs:
        tgt_text = tgt_text.replace(char, "")
    return tgt_text


def audio_processing(track_path, folder, audio, id, manifest):
    audio_wav = AudioSegment.from_file(track_path)
    mp3_path = track_path.replace(".wav", ".mp3").replace(root + "/" + folder, mp3_root)
    audio_wav.export(mp3_path)
    waveform, sample_rate = torchaudio.load(mp3_path)
    audio_name = audio.replace(".wav", "")
    npy_path = Path(root) / "fbank/" / f"{audio_name}.npy"
    extract_fbank_features(waveform, sample_rate, npy_path)
    track = AudioSegment.from_file(mp3_path)
    duration_ms = track.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["id"].append(audio)
    manifest["audio"].append(npy_path)
    text = text_processing(df["de"][int(id)])
    train_text.append(text)
    manifest["tgt_text"].append(text)


for folder in os.listdir(root):
    print(folder)
    if os.path.isdir(root + "/" + folder) and folder != "mp3" and folder != "fbank":
        counter = 0
        dialect = "ch_" + folder
        path = root + "/" + folder
        data_len = len([name for name in os.listdir(path)])
        print(data_len)
        split_size = data_len * 0.1
        print(split_size)
        counter_dev = 0
        counter_test = 0
        for audio in os.listdir(path):
            path_id = path + "/" + audio
            id = audio.replace(dialect + "_", "").replace(".wav", "")
            if counter % 2 == 0 and counter_test < split_size:
                audio_processing(path_id, folder, audio, int(id), test)
                counter_test = counter_test + 1
            if counter % 3 == 0 and counter_dev < split_size:
                audio_processing(path_id, folder, audio, int(id), dev)
                counter_dev = counter_dev + 1
            else:
                audio_processing(path_id, folder, audio, int(id), train)
            counter = counter + 1

df = pd.DataFrame.from_dict(train)
save_df_to_tsv(df, Path(root) / f"train_st_ch_de.tsv")
df = pd.DataFrame.from_dict(test)
save_df_to_tsv(df, Path(root) / f"test_st_ch_de.tsv")
df = pd.DataFrame.from_dict(dev)
save_df_to_tsv(df, Path(root) / f"dev_st_ch_de.tsv")

spm_filename_prefix = f"spm_char_st_ch_de"
# Generate config YAML
gen_config_yaml(
    Path(root),
    spm_filename_prefix + ".model",
    yaml_filename=f"config_st_ch_de.yaml",
    specaugment_policy="lb",
    )
# generating vocabulary
if len(train_text) > 0:
    gen_voc(train_text, spm_filename_prefix)

try:
    shutil.rmtree(root + "/mp3")
except OSError as e:
    print("Error: %s : %s" % (root + "/mp3", e.strerror))


# print(df.values[0])
# print(df["ch_vs"][:20])
# print(df["de"][:20])
#
# for col in data.columns:
#     print(col)
