#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import regex as re

from linastt.utils.kaldi import check_kaldi_dir, parse_kaldi_wavscp

def kaldi_split(
    input_folders,
    output_folder_base,
    splits_ratio=None,
    max_duration=False,
    throw_if_output_exists=True,
    ):

    if max_duration:
        assert not splits_ratio, "Cannot provide both max_duration and splits_ratio"
    else:
        assert splits_ratio, "Have to provide either max_duration or splits_ratio"
        assert min([isinstance(r, float) for r in splits_ratio]), "splits_ratio must be a list of floats"
        assert sum(splits_ratio) <= 1., "splits_ratio must sum to at most 1"
        assert min(splits_ratio) > 0., "splits_ratio must be all positive"

    # Make checks
    has_segments = None
    has_genders = None
    for input_folder in input_folders:

        for file in ["text", "wav.scp", "utt2dur", "spk2utt", "utt2spk"]:
            assert os.path.isfile(input_folder + "/" + file), f"Missing file: {input_folder}/{file}"


        if os.path.isfile(input_folder + "/segments"):
            _has_segments = True
        else:
            _has_segments = False
            print(f"WARNING: {input_folder} has no segments file")
        if has_segments is None:
            has_segments = _has_segments
        else:
            assert has_segments == _has_segments, f"Some folders have segments files and not others: {input_folder}"


        if os.path.isfile(input_folder + "/spk2gender"):
            _has_genders = True
        else:
            _has_genders = False
            print(f"WARNING: {input_folder} has no spk2gender file")
        if has_genders is None:
            has_genders = _has_genders
        else:
            assert has_genders == _has_genders, f"Some folders have spk2gender files and not others: {input_folder}"


        if throw_if_output_exists and os.path.isdir(output_folder_base):
            raise RuntimeError(f"Output folder already exists. Please remove it first if you want to regenerate it:\n#\trm -R {output_folder_base}")

    utt2wav = {} # Which audio file is each utt
    wav2dur = {} # How long is each wav

    for input_folder in input_folders:

        if has_segments:
            with open(input_folder + "/segments", 'r') as f:
                for line in f:
                    id, wav_id, _ = line.strip().split(" ", 2)
                    utt2wav[id] = wav_id

        with open(input_folder + "/utt2dur", 'r') as f:
            for line in f:
                id, duration = line.strip().split(" ")
                duration = float(duration)
                wav = utt2wav.get(id, id)
                wav2dur[wav] = wav2dur.get(wav, 0.) + float(duration)

    wav2split = {} # Which split is each wav
    if max_duration:
        isplit = 0
        accu_duration = 0.
        for i, (wav, duration) in enumerate(wav2dur.items()):
            accu_duration += duration
            if accu_duration > max_duration:
                if i > 0:
                    isplit += 1
                accu_duration = duration
            wav2split[wav] = isplit
    else:
        total_duration = sum(wav2dur.values())
        isplit = 0
        split_max_duration = splits_ratio[0] * total_duration
        accu_duration = 0
        for i, (wav, duration) in enumerate(wav2dur.items()):
            accu_duration += duration
            if accu_duration >= split_max_duration + 1e-3:
                total_accu_duration = sum(wav2dur[w] for w in wav2split.keys())
                if i > 0:
                    isplit += 1
                    if isplit >= len(splits_ratio):
                        print(f"WARNING: Ignoring next wav because it exceeded the given ratio, by {split_max_duration-accu_duration:.3f} seconds (taking {total_accu_duration/3600:.3f} hours / {total_duration/3600:.3f}).")
                        break
                accu_duration = duration
                remaining_ratio = sum(splits_ratio[isplit:])
                remaining_duration = total_duration - total_accu_duration
                split_max_duration = (splits_ratio[isplit]/remaining_ratio) * remaining_duration
            wav2split[wav] = isplit

    isplit = max(wav2split.values())

    splits = range(isplit+1)

    # Create the splits
    def get_split_folder(isplit):
        return output_folder_base + f"/split{isplit}"
    def create_split(isplit):
        output_folder = get_split_folder(isplit)

        os.makedirs(output_folder, exist_ok=True)
        return {
            "text": open(output_folder + "/text", 'a'),
            "wav.scp": open(output_folder + "/wav.scp", 'a'),
            "utt2dur": open(output_folder + "/utt2dur", 'a'),
            "utt2spk": open(output_folder + "/utt2spk", 'a'),
            "segments": open(output_folder + "/segments", 'a') if has_segments else None,
            "spk2gender": open(output_folder + "/spk2gender", 'a') if has_genders else None,
        }
    
    split2info = [create_split(i) for i in splits]

    # Helper to write the information in the right folder
    def get_split(id):
        wav = utt2wav.get(id, id)
        return wav2split.get(wav, None)
    def write_in_split(id, line, what):
        isplit = get_split(id)
        if isplit is None:
            return
        split = split2info[isplit]
        split[what].write(line)

    try:
        for input_folder in input_folders:

            with open(input_folder + "/text", 'r') as f:
                for i, line in enumerate(f):
                    id = _get_first_field(line)
                    write_in_split(id, line, "text")

            if has_segments:
                with open(input_folder + "/segments", 'r') as f:
                    for line in f:
                        id = _get_first_field(line)
                        write_in_split(id, line, "segments")

            with open(input_folder + "/wav.scp", 'r') as f:
                for line in f:
                    id = _get_first_field(line)
                    write_in_split(id, line, "wav.scp")

            with open(input_folder + "/utt2dur", 'r') as f:
                for line in f:
                    id = _get_first_field(line)
                    write_in_split(id, line, "utt2dur")


            spk2splits = {}
            with open(input_folder + "/utt2spk", 'r') as f:
                for line in f:
                    id = _get_first_field(line)
                    spk = line.strip().split(" ")[1]
                    isplit = get_split(id)
                    if isplit is not None:
                        if isplit not in spk2splits.get(spk, []):
                            spk2splits[spk] = spk2splits.get(spk, []) + [isplit]
                        write_in_split(id, line, "utt2spk")

            if os.path.isfile(input_folder + "/spk2gender"):
                with open(input_folder + "/spk2gender", 'r') as f:
                    for line in f:
                        spk = _get_first_field(line)
                        for isplit in spk2splits.get(spk, []):
                            split2info[isplit]["spk2gender"].write(line)

    finally:
        # Close files
        for split in split2info:
            for f in split.values():
                if f is None: continue
                f.close()

    for isplit in splits:
        check_kaldi_dir(get_split_folder(isplit))


def _get_first_field(line):
    f = line.split(" ", 1)
    assert len(f), "Got an empty line"
    return f[0]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Create a kaldi folder with only a subset of utterances from a kaldi folder")
    parser.add_argument("input_folder", type=str, help="Input folder(s) with kaldi files", nargs='+')
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--ratio", type=float, help="Wanted ratio. Example: 0.6 0.2 0.2", nargs="*")
    parser.add_argument("--max_duration", default=None, type=float, help="Maximum duration for a split (in seconds)")
    args = parser.parse_args()

    kaldi_split(
        args.input_folder, args.output_folder,
        splits_ratio=args.ratio,
        max_duration=args.max_duration,
    )
