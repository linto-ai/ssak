#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from linastt.utils.kaldi import parse_kaldi_wavscp
from linastt.utils.kaldi import check_kaldi_dir


def create_cut(input_folder, output_folder, n_first):

    utt_ids = []
    spk_ids = []

    for file in ["text", "wav.scp", "utt2dur", "spk2utt", "spk2gender"]:
        assert os.path.isfile(input_folder + "/" + file), f"No {file} in the input folder"

    if os.path.isfile(input_folder + "/segments"):
        raise NotImplementedError("Not implemented when the input folder contains a 'segments' file")

    os.makedirs(output_folder, exist_ok=True)

    with open(input_folder + "/text", 'r') as f, \
        open(output_folder + "/text", 'w') as text_file:
        for i, line in zip(range(n_first), f):
            utt_ids.append(line.split(" ")[0])
            text_file.write(line)

    with open(input_folder + "/wav.scp", 'r') as f, \
        open(output_folder + "/wav.scp", 'w') as wavscp_file:
        for line in f:
            id = line.split(" ")[0]
            if id in utt_ids:
                wavscp_file.write(line)

    with open(input_folder + "/utt2dur", 'r') as f, \
        open(output_folder + "/utt2dur", 'w') as utt2dur:
        lines = f.readlines()
        for line in lines:
            for id in utt_ids:
                if id in line.split(" "):
                    utt2dur.write(line)

    with open(input_folder + "/spk2utt", 'r') as f, \
        open(output_folder + "/spk2utt", 'w') as spk2utt:
        lines = f.readlines()
        for line in lines:
            spk = line.split(" ")[0]
            utt_s = line.split(" ")[1:]
            for id in utt_ids:
                if id in utt_s:
                    spk2utt.write(f"{spk} \t {id} \n")
                    spk_ids.append(spk)

    with open(input_folder + "/spk2gender", 'r') as f, \
        open(output_folder + "/spk2gender", 'w') as spk2gender:
        lines = f.readlines()
        for line in lines:
            for spk in spk_ids:
                if spk in line.split(" "):
                    spk2gender.write(line)

    return check_kaldi_dir(output_folder, language=None)


def _get_id(line):
    f = line.split(" ", 1)
    assert len(f), "Got an empty line"
    return f[0]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Creates a test version of a kaldi folder: taken n first lines")
    parser.add_argument("input_folder", type=str, help="Input folder with kaldi files")
    parser.add_argument("output_folder", type=str, help="Output folder to put the cut files")
    parser.add_argument("n_first", type=int, help="n first lines to keep")
    args = parser.parse_args()

    create_cut(args.input_folder, args.output_folder, args.n_first)
