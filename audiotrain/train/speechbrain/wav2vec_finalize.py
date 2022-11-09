import sys
import os
import shutil
import hyperpyyaml
import glob
import re
import speechbrain as sb

from audiotrain.infer.speechbrain_infer import speechbrain_load_model

def finalize_folder(
        folder,
        hparams_file = None,
    ):

    # Create a symbolic link to the last checkpoint with validation
    cpkt_folder = None
    for root, dirs, files in os.walk(folder):
        for d in sorted(dirs, reverse = True):
            if d.startswith("CKPT"):
                cpkt_folder = os.path.join(root, d)
                ckpt_file = os.path.join(cpkt_folder, "CKPT.yaml")
                if os.path.exists(ckpt_file):
                    with open(ckpt_file, "r") as f:
                        ckpt = hyperpyyaml.load_hyperpyyaml(f)
                        if "WER" in ckpt:
                            break
    assert cpkt_folder is not None, f"Could not find checkpoint folder in {folder}"
    final_folder = os.path.join(folder, "final")
    if os.path.exists(final_folder) or os.path.islink(final_folder):
        os.remove(final_folder)
    os.symlink(os.path.relpath(cpkt_folder, folder), final_folder)

    # Get parameters passed to the command line
    readme_file = os.path.join(folder, "README.txt")
    assert os.path.exists(readme_file), f"Could not find {readme_file}"
    args = ""
    with open(readme_file, "r") as f:
        for line in f:
            if ".py" in line:
                args = line.split()[1:]
                break
    _, _, overrides = sb.parse_arguments(args)

    # Get hyperparameters yaml file
    if hparams_file is None:
        for d in sorted(os.listdir(folder), reverse = True):
            if d.startswith("src"):
                files = glob.glob(os.path.join(folder, d, "*.yaml"))
                assert len(files) == 1, f"Found {len(files)} yaml files in {root}"
                hparams_file = files[0]
                break
    assert hparams_file is not None, f"Could not find hyperparams yaml file in {folder}"
    #hparams_in = easy_yaml_load(hparams_file)
    hparams_in = hyperpyyaml.load_hyperpyyaml(open(hparams_file), overrides = overrides)
    save_folder = hparams_in["save_folder"]

    # Create the hyperparams.yaml file
    keep_fields = [
        "sample_rate",
        "blank_index",
        "wav2vec2",
        "ctc_lin",
        "enc",
        "log_softmax",
    ]

    hparams_file_out = os.path.join(final_folder, "hyperparams.yaml")
    copied_fields = copy_yaml_fields(hparams_file, hparams_file_out, keep_fields, overrides = overrides)
    with open(hparams_file_out, "a") as f:
        f.write("""
tokenizer: !new:sentencepiece.SentencePieceProcessor

modules:
    encoder: !ref <encoder>

encoder: !new:speechbrain.nnet.containers.LengthsCapableSequential
    wav2vec2: !ref <wav2vec2>
    enc: !ref <enc>
    ctc_lin: !ref <ctc_lin>

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        wav2vec2: !ref <wav2vec2>
        enc: !ref <enc>
        ctc_lin: !ref <ctc_lin>
        tokenizer: !ref <tokenizer>

decoding_function: !name:speechbrain.decoders.ctc_greedy_decode
    blank_id: !ref <blank_index>
""")
        if "wav2vec2" not in copied_fields:
            f.write("""
wav2vec2: !new:speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
    source: LeBenchmark/wav2vec2-FR-7K-large
    save_path: !apply:audiotrain.utils.misc.get_cache_dir [ speechbrain/wav2vec2_checkpoint ] # ~/.cache/speechbrain/wav2vec2_checkpoint
""")

    # Copy the tokenizer
    tokenizer_in = None
    for filename in os.listdir(save_folder):
        if filename.endswith(".model") or filename == "tokenizer.ckpt":
            assert tokenizer_in is None, f"Found multiple tokenizers in {save_folder}"
            tokenizer_in = os.path.join(save_folder, filename)
    assert tokenizer_in is not None, f"Could not find tokenizer in {save_folder}"
    tokenizer_out = os.path.join(final_folder, "tokenizer.ckpt")
    shutil.copyfile(tokenizer_in, tokenizer_out)

    # Save the wav2vec2 model
    wav2vec_file = os.path.join(final_folder, "wav2vec2.ckpt")
    if not os.path.exists(wav2vec_file):
        if "wav2vec2" in hparams_in:
            wav2vec2 = hparams_in["wav2vec2"]
        else:
            model = speechbrain_load_model(hparams_in["base_model"])
            wav2vec2 = model.hparams.wav2vec2
        sb.utils.checkpoints.torch_save(wav2vec2, wav2vec_file)

def easy_yaml_load(filename):
    return hyperpyyaml.load_hyperpyyaml(open(filename), overrides = make_yaml_placeholder_overrides(filename))

def make_yaml_placeholder_overrides(yaml_file, default = "PLACEHOLDER"):
    """
    return a dictionary of overrides to be used with speechbrain
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None: return None
    override = {}
    with open(yaml_file, "r") as f:
        parent = None
        for line in f:
            if line == line.lstrip() and line != "" and ":" in line:
                field, value = line.split(":", 1)
                value = value.strip().split()
                if len(value):
                    value = value[0].strip()
                    if value == "!PLACEHOLDER":
                        override[field.strip()] = default
    return override
    
def copy_yaml_fields(from_file, to_file, fields, overrides = ""):
    def add_referenced_fields(line):
        if field in fields:
            for ref in re.findall(r"<[^<>]*>", line):
                ref = ref[1:-1]
                fields.append(ref)
    with open(from_file, "r") as f:
        content = {}
        for source in f, overrides.split("\n"):
            field = None
            for line in source:
                if line == line.lstrip() and line != "" and ":" in line:
                    field = line.split(":")[0].strip()
                    #assert field not in content, f"Duplicate field {field} in {from_file}"
                    content[field] = [line]
                    add_referenced_fields(line)
                elif line.rstrip().startswith("#"):
                    pass
                elif line.strip() != "":
                    assert field is not None, f"Unexpected line {line} in {from_file}"
                    content[field].append(line)
                    add_referenced_fields(line)
                else:
                    field = None
    copied_fields = []
    with open(to_file, "w") as f:
        for field, value, in content.items():
            if field in fields:
                copied_fields.append(field)
                f.write("".join(value)+"\n")
    return copied_fields

if __name__ == "__main__":
    
    if len(sys.argv) not in (2, 3):
        print("Usage: wav2vec_finalize.py <folder> <hparams>")
        sys.exit(1)

    finalize_folder(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
