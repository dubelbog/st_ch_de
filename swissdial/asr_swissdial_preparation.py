# coding=utf-8
from pydub import AudioSegment
from data_utils import save_df_to_tsv
from pathlib import Path
import pandas as pd
import os

root = "/Users/bdubel/Documents/ZHAW/BA/data/eth_ch_dialects"
json = "sentences_ch_de_transcribed.json"
data = pd.read_json(Path(root) / json)
df = pd.DataFrame(data)
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
test = {c: [] for c in MANIFEST_COLUMNS}
train = {c: [] for c in MANIFEST_COLUMNS}
dev = {c: [] for c in MANIFEST_COLUMNS}
ms = 1000
split = 0.1


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


def audio_processing(track_path, dialect, audio, id, manifest):
    audio = audio.replace(".wav", "")
    npy_path = Path(root) / "fbank/" / f"{audio}.npy"
    track = AudioSegment.from_file(track_path)
    duration_ms = track.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["id"].append(audio)
    manifest["audio"].append(npy_path)
    text = preprocess_transcript(df[dialect][int(id)])
    manifest["tgt_text"].append(text)


for folder in os.listdir(root):
    print(folder)
    if os.path.isdir(root + "/" + folder) and len(folder) == 2:
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
            if counter % 200 == 0:
                print(counter)
            path_id = path + "/" + audio
            id = audio.replace(dialect + "_", "").replace(".wav", "")
            if counter % 2 == 0 and counter_test < split_size:
                audio_processing(path_id, dialect, audio, int(id), test)
                counter_test = counter_test + 1
            if counter % 3 == 0 and counter_dev < split_size:
                audio_processing(path_id, dialect, audio, int(id), dev)
                counter_dev = counter_dev + 1
            else:
                audio_processing(path_id, dialect, audio, int(id), train)
            counter = counter + 1


df = pd.DataFrame.from_dict(train)
save_df_to_tsv(df, Path(root) / f"train_asr_dial.tsv")
df = pd.DataFrame.from_dict(test)
save_df_to_tsv(df, Path(root) / f"test_asr_dial.tsv")
df = pd.DataFrame.from_dict(dev)
save_df_to_tsv(df, Path(root) / f"dev_asr_dial.tsv")
