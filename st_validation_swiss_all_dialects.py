import os

# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/Clickworker_Test_Set/checkpoints/"
path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/parl_dial_asr/"

for file in os.listdir(path_checkpoints):
    if file.endswith(".pt"):
        subset = os.path.join(path_checkpoints, file)
        os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/parl_dial_asr "
                  "--config-yaml config_asr_de.yaml --gen-subset test_official --task speech_to_text "
                  "--path " + subset + " "
                  "--max-tokens 50000 --beam 5 --scoring sacrebleu")

