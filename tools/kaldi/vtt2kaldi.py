#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from linastt.utils.kaldi import check_kaldi_dir
from linastt.utils.text import format_special_characters


import os
import csv
import webvtt



def vtt2kaldi(meta_data_folder, input_file, audio_folder, output_folder, language=None):
    dataset_splits = ['clean_dev', 'clean_test', 'clean_train', 'noisy_dev', 'noisy_test', 'noisy_train']
    gender_map = {'Male': 'm', 'Female': 'f'}

    for dataset_split in dataset_splits:
        meta_file = os.path.join(meta_data_folder, f'{dataset_split}_meta.csv')
        dialects = set()

        with open(meta_file, 'r') as meta_csv:
            csv_reader = csv.DictReader(meta_csv)
            for row in csv_reader:
                dialects.add(row['dialect'])

        for dialect in dialects:
            dialect_output_folder = os.path.join(output_folder, dataset_split, dialect)
            os.makedirs(dialect_output_folder, exist_ok=True)

            with open(os.path.join(dialect_output_folder, 'utt2spk'), 'w') as utt2spk_f, \
                 open(os.path.join(dialect_output_folder, 'spk2utt'), 'w') as spk2utt_f, \
                 open(os.path.join(dialect_output_folder, 'text'), 'w') as text_f, \
                 open(os.path.join(dialect_output_folder, 'segments'), 'w') as segments_f, \
                 open(os.path.join(dialect_output_folder, 'wav.scp'), 'w') as wav_scp_f, \
                 open(os.path.join(dialect_output_folder, 'spk2gender'), 'w') as spk2gender_f, \
                 open(os.path.join(dialect_output_folder, 'utt2dur'), 'w') as utt2dur_f:

                with open(meta_file, 'r') as meta_csv:
                    csv_reader = csv.DictReader(meta_csv)
                    for _id, row in enumerate(csv_reader):
                        if row['dialect'] == dialect:
                            utt_id = f"MASC_{row['video_id']}-seg{_id:04d}"
                            audio_path = os.path.join(audio_folder, f"{row['video_id']}.wav")

                            gender = gender_map.get(row['gender'], random.choice(['m', 'f']))

                            utt2spk_f.write(f"{utt_id} {utt_id}\n")
                            spk2utt_f.write(f"{utt_id} {utt_id}\n")
                            wav_scp_f.write(f"{row['video_id']} sox {audio_path}  -t wav -r 16k -b 16 -c 1 - |\n")
                            spk2gender_f.write(f"{utt_id} {gender}\n")

                            for caption in webvtt.read(input_file + f"/{row['video_id']}.ar.vtt"):
                                text = caption.text
                                start = float(caption.start.strip().split(':')[-1])
                                end = float(caption.end.strip().split(':')[-1])
                                duration = end - start
                                text = format_special_characters(text)
                                text_f.write(f"{utt_id} {text}\n")
                                segments_f.write(f"{utt_id} {row['video_id']} {start:.2f} {end:.2f}\n")
                                utt2dur_f.write(f"{utt_id} {duration}\n")
                        
    

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Converts a dataset from  vtt format (like YouTube subtitles files) into kaldi format ')
    parser.add_argument('metadata_folder', help='input Meta Folder that contain the CSV files', type=str)
    parser.add_argument('input_subtitels',help='Input subtitels folder that conatin vtt files', type=str)
    parser.add_argument('audio_path', help='Path of Audio files', type=str)
    parser.add_argument('output_folder', help='Output kaldi folder', type=str)
    parser.add_argument('--language', default=None, help='Main Language', type=str)

    args = parser.parse_args()
    
    meta_data_folder = args.metadata_folder
    audio_folder = args.audio_path
    output_folder = args.output_folder
    input_subtitles = args.input_subtitels
    lang = args.language
    
    assert os.path.isdir(meta_data_folder), f"Input folder not found: {meta_data_folder}"
    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isdir(input_subtitles), f"Input file not found: {input_subtitles}"
    assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"
    
    vtt2kaldi(meta_data_folder, input_subtitles, audio_folder, output_folder, language=lang)