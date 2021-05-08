import os

def validation(root, checkpoints, config, manifest):
    for file in os.listdir(checkpoints):
        if file.endswith(".pt"):
            subset = os.path.join(checkpoints, file)
            os.system("python fairseq_cli_stchde/generate.py " + root +
                      " --config-yaml " + config +
                      " --gen-subset " + manifest +
                      " --task speech_to_text "
                      "--path " + subset + " --skip-invalid-size-inputs-valid-test "
                      "--max-tokens 50000 --beam 5 --scoring sacrebleu")


root_parl = "/home/ubuntu/data/st/parl/pert"
checkpoints_parl = "/home/ubuntu/data/st/parl/checkpoints/"
config = "config.yaml"
test = "config_minus15_5_08"

validation(root_parl, checkpoints_parl, config, test)




