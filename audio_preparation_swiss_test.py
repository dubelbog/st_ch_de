# coding=utf-8
from pydub import AudioSegment
import torchaudio
from data_utils import extract_fbank_features, save_df_to_tsv, gen_config_yaml, gen_vocab
from pathlib import Path
import pandas as pd
import shutil

# root_path = "/cluster/home/dubelbog/data/Swiss_Parliaments_Corpus/"
# local path
root_path = "/Users/bdubel/Documents/ZHAW/BA/data/Clickworker_Test_Set/"
root_path_data = root_path
path_manifest_swiss = root_path + "all.csv"
clip_path = root_path + "clips/"
mp3_path = root_path + "mp3/"
feature_root = Path(root_path)
suffix_flac = ".flac"
suffix_mp3 = ".mp3"
ms = 1000
task = "st_ch_de"

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


def manifest_preparation(manifest, track, track_path, file):
    waveform, sample_rate = torchaudio.load(track_path)
    utt_id = file.replace(".mp3", "")
    extract_fbank_features(waveform, sample_rate, feature_root / "fbank/" / f"{utt_id}.npy")
    manifest["id"].append(utt_id)
    npy_path = root_path + "fbank/" + utt_id + ".npy"
    manifest["audio"].append(npy_path)
    duration_ms = track.duration_seconds * ms
    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
    manifest["tgt_text"].append("ich bin der target text, der nicht stimmt!")


def audio_processing(file, manifest):
    file = file.replace("\n", "")
    track_path = mp3_path + file + suffix_mp3
    audio_file = clip_path + file
    audio_file = AudioSegment.from_file(audio_file)
    audio_file.export(track_path)
    track = AudioSegment.from_file(track_path)
    track.export(track_path)
    manifest_preparation(manifest, track, track_path, file)


def gen_voc(train_text, spm_filename_prefix):
    f = open(Path(root_path_data) / "test.txt", "a")
    for t in train_text:
        f.write(" ".join(t) + "\n")
    print(f.name)
    gen_vocab(
        Path(f.name),
        Path(root_path_data) / spm_filename_prefix
    )


def generate_manifest(split, manifest):
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, Path(root_path_data) / f"{split}_{task}.tsv")


def preparation():
    print("start")
    manifest_swiss = open(path_manifest_swiss, "r")
    data_len = len(open(path_manifest_swiss).readlines())
    test_manifest = {c: [] for c in MANIFEST_COLUMNS}
    train_text = []
    counter = 0
    for line in manifest_swiss:
        print(counter, " from ", data_len)
        if counter != 0:
            audio_processing(line, test_manifest)
        counter = counter + 1
        # generate manifest
    generate_manifest("test", test_manifest)
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

    try:
        shutil.rmtree(mp3_path)
    except OSError as e:
        print("Error: %s : %s" % (mp3_path, e.strerror))


preparation()
