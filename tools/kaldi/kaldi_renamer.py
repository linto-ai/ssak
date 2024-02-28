#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

def rename_audios_based_on_wavscp(input_folder, char_to_replace, replacement_char):
    with open(os.path.join(input_folder, "wav.scp"), 'r') as f:
        content = f.readlines()
        for line in tqdm(content):
            line = line.strip().split(" ", 1)
            file_path = line[1]
            new_file = file_path.split("/")[-1].replace(char_to_replace, replacement_char)
            new_file_path = os.path.join("/".join(file_path.split("/")[:-1]), new_file)
            os.rename(file_path, new_file_path)
            

def replace_char_in_file(file_path, char_to_replace, replacement_char, column=-1):
    new_content = ""
    with open(file_path, 'r') as f:
        content = f.readlines()
        for line in content:
            if column>=0:
                line = line.strip().split(" ", 1)
                line[column] = line[column].replace(char_to_replace, replacement_char)
                new_content += " ".join(line) + "\n"
            else:
                new_content += line.replace(char_to_replace, replacement_char)
    with open(file_path, 'w') as f:
        f.write(new_content)
    if os.path.exists(file_path):
        print(f"File '{file_path}' has been modified ('{char_to_replace}' -> '{replacement_char}')")
    else:
        raise Exception(f"File {file_path} missing")

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Replace a specific character by another one in all kaldi files")
    parser.add_argument("input_folder", type=str, help="Input folder with kaldi files")
    parser.add_argument("--char_to_replace", type=str, default=":", help="Character to replace")
    parser.add_argument("--replacement_char", type=str, default="-", help="Replacement character")
    args = parser.parse_args()
    
    if os.path.exists(os.path.join(args.input_folder, "utt2dur")):
        replace_char_in_file(os.path.join(args.input_folder, "utt2dur"), args.char_to_replace, args.replacement_char)
    if os.path.exists(os.path.join(args.input_folder, "utt2spk")):
        replace_char_in_file(os.path.join(args.input_folder, "utt2spk"), args.char_to_replace, args.replacement_char)
    if os.path.exists(os.path.join(args.input_folder, "wav.scp")):
        rename_audios_based_on_wavscp(args.input_folder, args.char_to_replace, args.replacement_char)
        replace_char_in_file(os.path.join(args.input_folder, "wav.scp"), args.char_to_replace, args.replacement_char)
    if os.path.exists(os.path.join(args.input_folder, "segments")):
        replace_char_in_file(os.path.join(args.input_folder, "segments"), args.char_to_replace, args.replacement_char, column=0)
    if os.path.exists(os.path.join(args.input_folder, "text")):
        replace_char_in_file(os.path.join(args.input_folder, "text"), args.char_to_replace, args.replacement_char, column=0)
    


# def rename_files(input_folder, char_to_replace, replacement_char, extension=".wav"):
#     files = os.listdir(input_folder)
#     for file in files:
#         if extension=="" or file.endswith(extension):
#             new_file = file.replace(char_to_replace, replacement_char)
#             os.rename(os.path.join(input_folder, file), os.path.join(input_folder, new_file))





