from pathlib import Path
from typing import Union
import csv
import pandas as pd


def get_evaluation_file_name(path):
    file_name = path
    file_name = file_name.rsplit('/', 1)[1]
    file_name = file_name.replace(".pt", ".tsv")
    checkpoint = file_name.replace("checkpoint", "")
    checkpoint = checkpoint.replace(".tsv", "")
    return file_name.replace("checkpoint", "evaluation"), checkpoint


def generate_manifest(manifest, root_path_data):
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, Path(root_path_data))


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

