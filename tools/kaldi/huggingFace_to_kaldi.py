#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
from tqdm import tqdm
from datasets import load_dataset
from linastt.utils.kaldi import check_kaldi_dir

def write_set(data, dir_out="Voxpopuli-fr", file_mode="W", speakers=dict(), id_spk=0):
    with open(os.path.join(dir_out,"text"), file_mode) as text_f, \
        open(os.path.join(dir_out,"utt2dur"), file_mode) as utt2dur_f, \
        open(os.path.join(dir_out,"wav.scp"), file_mode) as wav_f, \
        open(os.path.join(dir_out,"utt2spk"), file_mode) as utt2spk_f:
        for i, example in tqdm(enumerate(data), total=len(data)):
            duration = len(example['audio']['array'])/example['audio']['sampling_rate']
            spk_id = example['speaker_id']
            if spk_id is None or spk_id== "None":
                spk_id = f"spk-{example['audio_id']}"
            if spk_id not in speakers:
                speakers[spk_id] = f"{id_spk:05d}"
                id_spk += 1
            utt_id = speakers[spk_id]+"_"+example['audio_id']
            os.system(f"cp {example['audio']['path']} {os.path.join(dir_out,'wavs',example['audio_id'])}.wav")
            text_f.write(f"{utt_id} {example['raw_text']}\n")
            p = pathlib.Path(os.path.join(dir_out, "wavs", example['audio_id'])).resolve()
            wav_f.write(f"{utt_id} {p}.wav\n")
            utt2dur_f.write(f"{utt_id} {duration}\n")
            utt2spk_f.write(f"{utt_id} {speakers[spk_id]}\n")
    return speakers, id_spk


def write_dataset(huggingface_dataset, kaldi_dir="Voxpopuli-fr", language="fr", trust_remote_code=False, set_name=None):
    os.makedirs(os.path.join(kaldi_dir,"wavs"), exist_ok=True)
    speakers = dict()
    id_spk = 0

    if set_name is not None:
        data = load_dataset(huggingface_dataset, language, split=set_name, trust_remote_code=trust_remote_code)
        speakers, id_spk = write_set(data, kaldi_dir, "w", speakers, id_spk)
    else:
        data = load_dataset(huggingface_dataset, language, trust_remote_code=trust_remote_code)
        for split in tqdm(data.keys(), total=len(data.keys())):
            speakers, id_spk = write_set(data[split], kaldi_dir, "w" if id_spk==0 else "a", speakers, id_spk)

    check_kaldi_dir(kaldi_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert a dataset HuggingFace (working at least for VoxPopuli) to Kaldi")
    parser.add_argument("--dataset", type=str, default="facebook/voxpopuli",help="Dataset name")
    parser.add_argument("--kaldi_path", type=str, default="Voxpopuli_fr",help="Path to new Kaldi directory")
    parser.add_argument("--language", type=str, default="fr",help="Language of the dataset")
    parser.add_argument("--trust_remote_code", action="store_false",help="Trust the remote code to run code locally. Default is False.")
    parser.add_argument("--set_name", type=str, default=None, help="Name of the set to convert (train, test, validation). If None, all sets are converted.")
    args = parser.parse_args()

    write_dataset(huggingface_dataset=args.dataset, kaldi_dir=args.kaldi_path, language=args.language, trust_remote_code=args.trust_remote_code, set_name=args.set_name)