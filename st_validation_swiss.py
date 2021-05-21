import os

# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/eth_ch_dialects/checkpoints/mix"
# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swiss_all/checkpoints/average"
# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swiss_09/average"
# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/checkpoints_st_asr"
# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/checkpoints_st_all"
# path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/asr_parl_dial/checkpoints"
path_checkpoints = "/Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/checkpoints_st_all/best"

# for file in os.listdir(path_checkpoints):
#     if file.endswith(".pt"):
#         subset = os.path.join(path_checkpoints, file)
#         os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/swiss_all/perturbation "
#                   "--config-yaml config_st_ch_de.yaml --gen-subset train_perturbation --task speech_to_text "
#                   "--path " + subset + " "
#                   "--max-tokens 50000 --beam 5 --scoring sacrebleu")


# change encoder path back in file /Users/bdubel/Documents/ZHAW/BA/st_ch_de/fairseq_stchde/checkpoint_utils.py line 669
for file in os.listdir(path_checkpoints):
    if file.endswith(".pt"):
        subset = os.path.join(path_checkpoints, file)
        os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/asr_parl_dial "
                  "--config-yaml config_asr_de.yaml --gen-subset test_st_ch_de_dial_local --task speech_to_text "
                  "--path " + subset + " "
                  "--max-tokens 50000 --beam 5 --scoring sacrebleu")
