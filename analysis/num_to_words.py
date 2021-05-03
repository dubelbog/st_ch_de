import pandas as pd
from pathlib import Path
from num2words import num2words
from data_utils import save_df_to_tsv


file = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/train_st_ch_de_parl.tsv"
file_openstack = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/train_st_ch_de_parl_open.tsv"
file_cluster = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/train_st_ch_de_parl_num2w.tsv"
pred = pd.read_table(file)
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
manifest_openstack = {c: [] for c in MANIFEST_COLUMNS}
manifest_cluster = {c: [] for c in MANIFEST_COLUMNS}


def num_2_words(prediction, counter):
    for word in prediction.split(" "):
        if word.isdigit():
            print(prediction)
            counter = counter + 1
            prediction = prediction.replace(" " + word + " ", " " + num2words(word, lang='de') + " ")
            prediction = prediction.replace(word + " ", num2words(word, lang='de') + " ")
            prediction = prediction.replace(" " + word, " " + num2words(word, lang='de'))
            print(prediction)
    return prediction, counter


def write_manifest(manifest, id, audio, n_frames, tgt_text):
    manifest["id"].append(id)
    manifest["audio"].append(audio)
    manifest["n_frames"].append(n_frames)
    manifest["tgt_text"].append(tgt_text)


def generate_manifest(manifest, path):
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, Path(path))


def iterate_line(doc):
    counter = 0
    tot = doc.shape[0]
    for line in range(tot):
        print(line, " von ", tot)
        target, counter = num_2_words(str(doc.values[line][3]), counter)
        write_manifest(manifest_cluster, doc.values[line][0], doc.values[line][1], doc.values[line][2], target)
        path_openstack = str(doc.values[line][1]).\
            replace("/cluster/home/dubelbog/data/Swiss_Parliaments_Corpus", "/home/ubuntu/data/st/parl")
        write_manifest(manifest_openstack, doc.values[line][0], path_openstack, doc.values[line][2], target)
    print(counter)


iterate_line(pred)
generate_manifest(manifest_cluster, file_cluster)
generate_manifest(manifest_openstack, file_openstack)



