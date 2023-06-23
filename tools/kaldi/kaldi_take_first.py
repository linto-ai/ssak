#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from linastt.utils.kaldi import parse_kaldi_wavscp
from linastt.utils.kaldi import check_kaldi_dir


def create_cut(input_folder, output_folder, n_first):

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    utt_ids = []
    _ids = []
    spk_ids = []

    for file in files:
        assert os.path.isfile(input_folder + file), f"No {file} in the input folder"

    with open(input_folder + "/" + "text", 'r') as f:
        new_text = []
        cut = [next(f) for _ in range(n_first)]
        for line in cut:
            utt_ids.append(line.split(" ")[0])
            new_text.append(line)
        with open(output_folder + "/" + "text", 'w') as text_file:
            text_file.writelines(new_text)

    parsed_wav = parse_kaldi_wavscp(input_folder + "/" + "wav.scp")

    with open(output_folder + '/wav.scp', 'w') as wavscp_file:
        for i in utt_ids:
            if i in parsed_wav.keys():
                wavscp_file.write(i + " sox " + os.path.abspath(parsed_wav[i]) + " -t wav -r 16k -b 16 -c 1 - |\n")

    with open(input_folder + "/" + "utt2dur", 'r') as f:
        with open(output_folder + '/utt2dur', 'w') as utt2dur:
            lines = f.readlines()
            for line in lines:
                for id in utt_ids:
                    if id in line.split(" "):
                        utt2dur.write(line)

    with open(input_folder + "/" + "spk2utt", 'r') as f:
        with open(output_folder + '/spk2utt', 'w') as spk2utt:
            lines = f.readlines()
            for line in lines:
                spk = line.split(" ")[0]
                utt_s = line.split(" ")[1:]
                for id in utt_ids:
                    if id in utt_s:
                        spk2utt.write(f"{spk} \t {id} \n")
                        spk_ids.append(spk)

    with open(input_folder + "/" + "spk2gender", 'r') as f:
        with open(output_folder + '/spk2gender', 'w') as spk2gender:
            lines = f.readlines()
            for line in lines:
                for spk in spk_ids:
                    if spk in line.split(" "):
                        spk2gender.write(line)

    return check_kaldi_dir(output_folder, language=None)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Creates a test version of a kaldi folder: taken n first lines")
    parser.add_argument("input_folder", type=str, help="Input folder with kaldi files")
    parser.add_argument("output_folder", type=str, help="Output folder to put the cut files")
    parser.add_argument("n_first", type=int, help="n first lines to keep")
    args = parser.parse_args()

    create_cut(args.input_folder, args.output_folder, args.n_first)
