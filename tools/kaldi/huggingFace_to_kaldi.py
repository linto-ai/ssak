#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np
import pathlib



def write_set(data, file_mode="W", speakers=dict(), id_spk=0):
    # it works for Voxpopuli
    with open(os.path.join(args.kaldi_path,"text"), file_mode) as text_f, \
        open(os.path.join(args.kaldi_path,"utt2dur"), file_mode) as utt2dur_f, \
        open(os.path.join(args.kaldi_path,"wav.scp"), file_mode) as wav_f, \
        open(os.path.join(args.kaldi_path,"utt2spk"), file_mode) as utt2spk_f:
        for i, example in tqdm(enumerate(data), total=len(data)):
            duration = len(example['audio']['array'])/example['audio']['sampling_rate']
            spk_id = example['speaker_id']
            if spk_id is None or spk_id== "None":
                spk_id = f"spk-{example['audio_id']}"
            if spk_id not in speakers:
                speakers[spk_id] = f"{id_spk:05d}"
                id_spk += 1
            utt_id = speakers[spk_id]+"_"+example['audio_id']
            os.system(f"cp {example['audio']['path']} {os.path.join(args.kaldi_path,'wavs',example['audio_id'])}.wav")
            text_f.write(f"{utt_id} {example['raw_text']}\n")
            p = pathlib.Path(os.path.join(args.kaldi_path, "wavs", example['audio_id'])).resolve()
            wav_f.write(f"{utt_id} {p}.wav\n")
            utt2dur_f.write(f"{utt_id} {duration}\n")
            utt2spk_f.write(f"{utt_id} {speakers[spk_id]}\n")
    return speakers, id_spk

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert a dataset HuggingFace (VoxPopuli) to Kaldi")
    parser.add_argument("--dataset", type=str, default="facebook/voxpopuli",help="Dataset name")
    parser.add_argument("--kaldi_path", type=str, default="Voxpopuli_fr",help="Path to new Kaldi directory")
    args = parser.parse_args()

    os.makedirs(args.kaldi_path+"/wavs", exist_ok=True)
    
    speakers = dict()
    id_spk = 0

    voxpopuli_fr = load_dataset(args.dataset, "fr", trust_remote_code=True, split="test")
    print(voxpopuli_fr[0])
    speakers, id_spk = write_set(voxpopuli_fr, "w", speakers, id_spk)

    voxpopuli_fr = load_dataset(args.dataset, "fr", trust_remote_code=True, split="validation")
    speakers, id_spk = write_set(voxpopuli_fr, "a", speakers, id_spk)

    voxpopuli_fr = load_dataset(args.dataset, "fr", trust_remote_code=True, split="train")
    speakers, id_spk =  write_set(voxpopuli_fr, "a", speakers, id_spk)
