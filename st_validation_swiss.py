import os

path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swiss_09/checkpoints/"

for file in os.listdir(path_checkpoints):
    if file.endswith(".pt"):
        subset = os.path.join(path_checkpoints, file)
        os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/swiss_09 "
                  "--config-yaml config_st_ch_de.yaml --gen-subset test_st_ch_de --task speech_to_text "
                  "--path " + subset + " "
                  "--max-tokens 50000 --beam 5 --scoring sacrebleu")

