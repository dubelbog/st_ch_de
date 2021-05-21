import os

# "--path /Users/bdubel/Documents/ZHAW/BA/data/end_models/parl_archimob_asr.pt:/Users/bdubel/Documents/ZHAW/BA/data/end_models/parl_dial_asr_ch.pt:/Users/bdubel/Documents/ZHAW/BA/data/end_models/avg_parl_dial_encoder.pt "
os.system("python fairseq_cli_stchde/generate.py /Users/bdubel/Documents/ZHAW/BA/data/swissdial_parl/parl_dial_asr "
          "--config-yaml config_asr_de.yaml --gen-subset test_official --task speech_to_text "
          "--path /Users/bdubel/Documents/ZHAW/BA/data/end_models/tuning_parl_arch.pt:/Users/bdubel/Documents/ZHAW/BA/data/end_models/tuning_parl_dial_ch.pt "
          "--max-tokens 50000 --beam 5 --scoring sacrebleu")

