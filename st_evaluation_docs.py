import os
import pandas as pd
from fairseq_cli_stchde.helper_utils import generate_manifest

path_eval = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss"
eval_docs = []
checkpoints = []
MANIFEST_COLUMNS = ["id", "target_txt", "checkpoint", "prediction"]
eval_manifest = {c: [] for c in MANIFEST_COLUMNS}

for file in os.listdir(path_eval):
    if file.endswith(".tsv"):
        subset = os.path.join(path_eval, file)
        df = pd.read_table(subset)
        eval_docs.append(df)
        checkpoints.append(file.removesuffix(".tsv").removeprefix("evaluation"))

for text in range(eval_docs[0].shape[0]):
    id = eval_docs[0].values[text][0]
    eval_manifest["id"].append(id)
    target_txt = eval_docs[0].values[text][1]
    eval_manifest["target_txt"].append(target_txt)
    eval_manifest["checkpoint"].append("-")
    eval_manifest["prediction"].append("-")
    for doc in range(len(eval_docs)):
        checkpoint = checkpoints[doc]
        eval_manifest["checkpoint"].append(checkpoint)
        eval_manifest["target_txt"].append(target_txt)
        eval_manifest["id"].append(id)
        prediction = eval_docs[doc].values[text][2]
        eval_manifest["prediction"].append(prediction)

generate_manifest(eval_manifest, "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/swiss/overview/st_progress.tsv")

