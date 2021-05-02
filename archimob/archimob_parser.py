import xml.etree.ElementTree as ET
import os
import pandas as pd
from data_utils import extract_fbank_features, save_df_to_tsv
from pathlib import Path
import torchaudio
from pydub import AudioSegment

COLUMNS_PUBLIC = ["id", "tgt_text_de", "tgt_text_ch", "speaker"]
COLUMNS_LOCAL = ["id", "audio", "n_frames", "tgt_text", "speaker"]
manifest_public = {c: [] for c in COLUMNS_PUBLIC}
manifest_local_ch = {c: [] for c in COLUMNS_LOCAL}
manifest_local_de = {c: [] for c in COLUMNS_LOCAL}
manifest_openstack_ch = {c: [] for c in COLUMNS_LOCAL}
manifest_openstack_de = {c: [] for c in COLUMNS_LOCAL}

root_path = "/Users/bogumiladubel/Documents/BA/data/st/archimob/XML_2/Archimob_Release_2/"
audio_path = "/Users/bogumiladubel/Documents/BA/data/st/archimob/audio_segmented_anonymized/"
audio_path_openstack = "/home/ubuntu/data/st/archimob/audio_segmented_anonymized/"
root_path_data = "/Users/bogumiladubel/Documents/BA/repos/st_ch_de/archimob"
feature_root = "/Users/bogumiladubel/Documents/BA/data/st/archimob/fbank"


def helper_public_manifest(id, de, ch, speaker):
    manifest_public["id"].append(id)
    manifest_public["tgt_text_de"].append(de)
    manifest_public["tgt_text_ch"].append(ch)
    manifest_public["speaker"].append(speaker)


def manifest_writer(manifest, id, audio, n_frames, tgt_text, speaker):
    manifest["id"].append(id)
    manifest["audio"].append(audio)
    manifest["n_frames"].append(n_frames)
    manifest["tgt_text"].append(tgt_text)
    manifest["speaker"].append(speaker)


def convert_track(path, file_name):
    npy_path_openstack = ""
    npy_path_local = ""
    frames = 0
    if os.path.exists(path):
        waveform, sample_rate = torchaudio.load(path)
        utt_id = str(file_name.replace(".wav", ""))
        extract_fbank_features(waveform, sample_rate, Path(feature_root) / f"{utt_id}.npy")
        npy_path_openstack = "/home/ubuntu/data/st/archimob/fbank/" + utt_id + ".npy"
        npy_path_local = feature_root + "/" + utt_id + ".npy"
        track = AudioSegment.from_file(path)
        duration_ms = track.duration_seconds * 1000
        frames = int(1 + (duration_ms - 25) / 10)
    return npy_path_local, npy_path_openstack, frames


ALLOWED_CHARS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ä', 'ö', 'ü',
    ' '
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


def parse_xml(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    for element in root:
        if element.tag.split("}")[1] == "text":
            for body in element:
                if body.tag.split("}")[1] == "body":
                    for utt in body:
                        file_name = utt.attrib['start'].split("#")[1].replace("-", "_") + ".wav"
                        folder = file_name.split("_T")[0].replace("d", "")
                        person = utt.attrib['who']
                        if person.startswith("person_db#"):
                            person = person.split("#")[1]
                        id = folder + "/" + file_name
                        path = audio_path + id
                        npy_path_local, npy_path_openstack, n_frames = convert_track(path, file_name)
                        ch_text = []
                        de_text = []
                        for word in utt:
                            if word.tag.split("}")[1] == "w":
                                if word.text:
                                    ch_text.append(word.text)
                                if word.attrib['normalised'] != "==":
                                    de_text.append(word.attrib['normalised'])
                        if len(ch_text) != 0 and len(de_text) != 0 and npy_path_local != "" and n_frames != 0:
                            ch = preprocess_transcript(' '.join(ch_text))
                            de = preprocess_transcript(' '.join(de_text))
                            manifest_writer(manifest_local_ch, id, npy_path_local, n_frames, ch, person)
                            manifest_writer(manifest_local_de, id, npy_path_local, n_frames, de, person)
                            manifest_writer(manifest_openstack_ch, id, npy_path_openstack, n_frames, ch, person)
                            manifest_writer(manifest_openstack_de, id, npy_path_openstack, n_frames, de, person)
                            helper_public_manifest(id, de, ch, person)


counter = 0
for file in os.listdir(root_path):
    if file.endswith(".xml"):
        counter = counter + 1
        print("file nr: ", counter, " : ", file)
        parse_xml(root_path + file)


def save_manifest(file, manifest):
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, Path(root_path_data) / f"{file}.tsv")


save_manifest("archimob", manifest_public)
save_manifest("archimob_local_de", manifest_local_de)
save_manifest("manifest_local_ch", manifest_local_ch)
save_manifest("manifest_openstack_ch", manifest_openstack_ch)
save_manifest("manifest_openstack_de", manifest_openstack_de)

