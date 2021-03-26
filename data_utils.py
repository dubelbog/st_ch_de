import torch
from typing import Optional, Union, List, Dict, Any
import numpy as np
from pathlib import Path
import csv
import sentencepiece as sp

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


# start fairseq code
def _get_kaldi_fbank(
        waveform: np.ndarray, sample_rate: int, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform.squeeze()), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(
        waveform: np.ndarray, sample_rate, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi
        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def _convert_to_mono(
        waveform: torch.FloatTensor, sample_rate: int
) -> torch.FloatTensor:
    if waveform.shape[0] > 1:
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError(
                "Please install torchaudio to convert multi-channel audios"
            )
        effects = [['channels', '1']]
        return ta_sox.apply_effects_tensor(waveform, sample_rate, effects)[0]
    return waveform


def gen_vocab(
        input_path: Path, output_path_prefix: Path, model_type="bpe",
        vocab_size=1000, special_symbols: Optional[List[str]] = None
):
    # Train SentencePiece Model
    # todo: set the correct number of cpu's line 81
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--hard_vocab_limit=false",
        "--character_coverage=1.0",
        f"--num_threads={8}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
            vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
            and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
            and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
            and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix.as_posix() + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def extract_fbank_features(
        waveform: torch.FloatTensor,
        sample_rate: int,
        output_path: Optional[Path] = None,
        n_mel_bins: int = 80,
        overwrite: bool = False,
):
    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform = _convert_to_mono(waveform, sample_rate)
    _waveform = _waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform.numpy()

    features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable fbank feature extraction"
        )

    if output_path is not None:
        np.save(output_path.as_posix(), features)
    else:
        return features


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

def gen_config_yaml(
        manifest_root: Path,
        spm_filename: str,
        yaml_filename: str = "config.yaml",
        specaugment_policy: str = "lb",
        prepend_tgt_lang_tag: bool = False,
        sampling_alpha: float = 1.0,
        audio_root: str = "",
        cmvn_type: str = "utterance",
        gcmvn_path: Optional[Path] = None,
):
    manifest_root = manifest_root.absolute()
    writer = S2TDataConfigWriter(manifest_root / yaml_filename)
    writer.set_vocab_filename(spm_filename.replace(".model", ".txt"))
    writer.set_input_channels(1)
    writer.set_input_feat_per_channel(80)
    specaugment_setters = {
        "lb": writer.set_specaugment_lb_policy,
        "ld": writer.set_specaugment_ld_policy,
        "sm": writer.set_specaugment_sm_policy,
        "ss": writer.set_specaugment_ss_policy,
    }
    specaugment_setter = specaugment_setters.get(specaugment_policy, None)
    if specaugment_setter is not None:
        specaugment_setter()
    writer.set_bpe_tokenizer(
        {
            "bpe": "sentencepiece",
            "sentencepiece_model": (manifest_root / spm_filename).as_posix(),
        }
    )
    if prepend_tgt_lang_tag:
        writer.set_prepend_tgt_lang_tag(True)
    writer.set_sampling_alpha(sampling_alpha)

    if cmvn_type not in ["global", "utterance"]:
        raise NotImplementedError

    writer.set_feature_transforms("_train", [f"{cmvn_type}_cmvn", "specaugment"])
    writer.set_feature_transforms("*", [f"{cmvn_type}_cmvn"])

    if cmvn_type == "global":
        assert gcmvn_path is not None, (
            'Please provide path of global cmvn file.'
        )
        writer.set_global_cmvn(gcmvn_path)

    if len(audio_root) > 0:
        writer.set_audio_root(audio_root)
    writer.flush()


def gen_voc(train_text, spm_filename_prefix):
    f = open(Path("../data/sound").absolute() / "test.txt", "a")
    for t in train_text:
        f.write(" ".join(t.split()[0:4]) + "\n")
    print(f.name)
    gen_vocab(
        Path(f.name),
        Path("../data/sound") / spm_filename_prefix
    )


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1

    def __init__(self, yaml_path: Path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML for S2T data config YAML files")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename: str = "dict.txt"):
        self.config["vocab_filename"] = vocab_filename

    def set_specaugment(
            self,
            time_wrap_w: int,
            freq_mask_n: int,
            freq_mask_f: int,
            time_mask_n: int,
            time_mask_t: int,
            time_mask_p: float,
    ):
        self.config["specaugment"] = {
            "time_wrap_W": time_wrap_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    def set_specaugment_lb_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=1,
            freq_mask_f=27,
            time_mask_n=1,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_ld_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_sm_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=15,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_specaugment_ss_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_input_channels(self, input_channels: int = 1):
        self.config["input_channels"] = input_channels

    def set_input_feat_per_channel(self, input_feat_per_channel: int = 80):
        self.config["input_feat_per_channel"] = input_feat_per_channel

    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_global_cmvn(self, stats_npz_path: str):
        self.config["stats_npz_path"] = stats_npz_path

    def set_feature_transforms(self, split: str, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms

    def set_prepend_tgt_lang_tag(self, flag: bool = True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_sampling_alpha(self, sampling_alpha: float = 1.0):
        self.config["sampling_alpha"] = sampling_alpha
# end fairseq code
