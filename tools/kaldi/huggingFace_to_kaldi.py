#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import shutil
import re
from tqdm import tqdm
from datasets import load_dataset
from linastt.utils.kaldi import check_kaldi_dir

def write_set(data, dir_out="Voxpopuli-fr", file_mode="W", speakers=dict(), id_spk=0, missing_raw_replacement=None):
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
            audio_id = example['audio_id'].replace(":", "-")
            utt_id = speakers[spk_id]+"_"+audio_id
            if not os.path.exists(os.path.join(dir_out, "wavs", audio_id)+".wav"):
                shutil.copy2(example['audio']['path'], os.path.join(dir_out,'wavs', audio_id)+".wav")
            if example['raw_text']=="" and missing_raw_replacement is not None and example['audio_id'] in missing_raw_replacement:
                replacement = missing_raw_replacement[example['audio_id']]
                replacement = re.sub(r"0{9}\b", " milliards", replacement)
                replacement = re.sub(r"0{6}\b", " millions", replacement)
                text_f.write(f"{utt_id} {replacement}\n")
            else:
                text_f.write(f"{utt_id} {example['raw_text']}\n")
            p = pathlib.Path(os.path.join(dir_out, "wavs", audio_id)).resolve()
            wav_f.write(f"{utt_id} {p}.wav\n")
            utt2dur_f.write(f"{utt_id} {duration}\n")
            utt2spk_f.write(f"{utt_id} {speakers[spk_id]}\n")
    return speakers, id_spk


def write_missings(data, dir_out):
    nb_checks = 100
    ct_checks = 0
    ct = 0
    with open(os.path.join(dir_out,"normalized_missing_raw"), "w") as normalized_missing_raw, \
        open(os.path.join(dir_out,"normalized_check"), "w") as normalized_check, \
        open(os.path.join(dir_out,"raw_check"), "w") as raw_check:
        for set_name in data.keys():
            for i, example in tqdm(enumerate(data[set_name]), total=len(data[set_name]), desc=f"Set {set_name}"):
                if example['raw_text']=="":
                    normalized_missing_raw.write(f"{example['audio_id']} {example['normalized_text']}\n")
                    ct+=1
                else:
                    if ct_checks<=nb_checks:
                        normalized_check.write(f"{example['audio_id']} {example['normalized_text']}\n")
                        raw_check.write(f"{example['audio_id']} {example['raw_text']}\n")
    print(f"Number of missing raw text: {ct}")
         
                    
def write_dataset(huggingface_dataset, kaldi_dir="Voxpopuli-fr", language="fr", trust_remote_code=False, set_name=None, missing_raw_replacement_file=None):
    os.makedirs(os.path.join(kaldi_dir,"wavs"), exist_ok=True)
    speakers = dict()
    id_spk = 0
    missing_raw_replacement = None
    if missing_raw_replacement_file is not None:
        with open(missing_raw_replacement_file, "r") as f:
            missing_raw_replacement = f.readlines()
            missing_raw_replacement = [x.strip() for x in missing_raw_replacement]
            missing_raw_replacement = {x.split(" ",1)[0]: x.split(" ",1)[1] for x in missing_raw_replacement}
    if set_name is not None:
        data = load_dataset(huggingface_dataset, language, split=set_name, trust_remote_code=trust_remote_code)
        speakers, id_spk = write_set(data, kaldi_dir, "w", speakers, id_spk, missing_raw_replacement=missing_raw_replacement) 
    else:
        data = load_dataset(huggingface_dataset, language, trust_remote_code=trust_remote_code)
        for split in tqdm(data.keys(), total=len(data.keys())):
            speakers, id_spk = write_set(data[split], kaldi_dir, "w" if id_spk==0 else "a", speakers, id_spk, missing_raw_replacement=missing_raw_replacement)

    check_kaldi_dir(kaldi_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert a dataset HuggingFace (working at least for VoxPopuli) to Kaldi")
    parser.add_argument("--dataset", type=str, default="facebook/voxpopuli",help="Dataset name")
    parser.add_argument("--kaldi_path", type=str, default="Voxpopuli_fr",help="Path to new Kaldi directory")
    parser.add_argument("--language", type=str, default="fr",help="Language of the dataset")
    parser.add_argument("--trust_remote_code", action="store_false",help="Trust the remote code to run code locally. Default is False.")
    parser.add_argument("--set_name", type=str, default=None, help="Name of the set to convert (train, test, validation). If None, all sets are converted.")
    parser.add_argument("--missing_raw_replacement_file", type=str, default="/home/abert/Linagora/datasets/raw_missing_raw", help="")

    args = parser.parse_args()

    write_dataset(huggingface_dataset=args.dataset, kaldi_dir=args.kaldi_path, language=args.language, trust_remote_code=args.trust_remote_code, set_name=args.set_name, missing_raw_replacement_file=args.missing_raw_replacement_file)
    # write_missings(load_dataset(args.dataset, args.language, trust_remote_code=args.trust_remote_code), "")