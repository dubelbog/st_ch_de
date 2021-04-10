# coding=utf-8
from pydub import AudioSegment
import torchaudio
from data_utils import extract_fbank_features, save_df_to_tsv, gen_config_yaml, gen_vocab
from pathlib import Path
import pandas as pd
import shutil

root_path = "/cluster/home/dubelbog/data/comon_voice_de/"
# local path
# root_path = "/Users/bdubel/Documents/ZHAW/BA/data/comon_voice_de/"
root_path_data = root_path
path_manifest_swiss = root_path + "test.tsv"
clip_path = root_path + "clips/"
feature_root = Path(root_path) / "fbank"
suffix_mp3 = ".mp3"
ms = 1000
task = "st_de_de"

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

def print_audio_infos(track):
    print("frame_rate", track.frame_rate)
    print("duration_seconds", track.duration_seconds)
    print("sample_width", track.sample_width)
    print("frame_width", track.frame_width)
    print("frame_count", track.frame_count())


def text_processing(data):
    tgt_text_arr = data[2:len(data) - 3]
    tgt_text = " ".join(tgt_text_arr)
    tgt_text = tgt_text.replace('ß', 'ss')
    tgt_text = tgt_text.lower()
    replace_signs = ['-', '–', "»", "«", ".", ",", "(", ")", "?", "!", "/", ":", ";", "]", "["]
    for char in replace_signs:
        tgt_text = tgt_text.replace(char, "")
    return tgt_text


def manifest_preparation(manifest, track, data, tgt_text, track_path):
    waveform, sample_rate = torchaudio.load(track_path)
    utt_id = data[1].replace(".mp3", "")
    extract_fbank_features(waveform, sample_rate, feature_root / f"{utt_id}.npy")
    manifest["id"].append(utt_id)
    manifest["audio"].append(feature_root / f"{utt_id}.npy")
    duration_ms = track.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["tgt_text"].append(tgt_text)
    manifest["speaker"].append(data[0])


def audio_processing(data, manifest, tgt_text):
    file = data[1]
    audio_file_path = clip_path + file
    audio_file = AudioSegment.from_file(audio_file_path)
    manifest_preparation(manifest, audio_file, data, tgt_text, audio_file_path)


def gen_voc(train_text, spm_filename_prefix):
    f = open(Path(root_path_data) / "test.txt", "a")
    for t in train_text:
        f.write(" ".join(t) + "\n")
    print(f.name)
    gen_vocab(
        Path(f.name),
        Path(root_path_data) / spm_filename_prefix
    )


def helper_preparation(line, train_text, manifest):
    data = line.split()
    tgt_text = text_processing(data)
    print(tgt_text)
    train_text.append(tgt_text)
    audio_processing(data, manifest, tgt_text)


def generate_manifest(split, manifest):
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, Path(root_path_data) / f"{split}_{task}.tsv")


def preparation():
    print("start")
    manifest_de = open(path_manifest_swiss, "r")
    data_len = len(open(path_manifest_swiss).readlines())
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    train_text = []
    counter = 0
    for line in manifest_de:
        print(counter, " from ", data_len)
        if counter != 0:
            helper_preparation(line, train_text, manifest)
        counter = counter + 1
        # generate manifest
    generate_manifest("test", manifest)
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


preparation()
