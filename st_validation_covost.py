import os

path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/covost/st_sv/"

for file in os.listdir(path_checkpoints):
    if file.endswith(".pt"):
        subset = os.path.join(path_checkpoints, file)
        os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/covost/sv-SE "
                  "--config-yaml config_st_sv-SE_en.yaml --gen-subset test_st_sv-SE_en --task speech_to_text "
                  "--path " + subset + " "
                  "--max-tokens 50000 --beam 5 --scoring sacrebleu")

