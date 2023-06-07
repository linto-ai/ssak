#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linastt.utils.kaldi import check_kaldi_dir

import os
import csv

def generate_examples(filepath, path_to_clips, max_existence_file_check=10):
    """
    Yields examples as dictionaries
    {
        "filename": ...,
        "text": ..., 
        "gender": ..., # "m" / "f" / "other" / ""
        "client_id": ..., # unique id for each speaker / sentence
    }

    filepath: path to the TSV or CSV file
    path_to_clips: path to the folder containing the audio files
    max_existence_file_check: maximum number of files to check if they exist (the first ones)
    """

    # data_fields =     ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"]
    # data_fields_old = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"]
    # data_fields_csv = ["filename", "text", "up_votes", "down_votes", "age", "gender", "accent", "duration"]
    is_csv = filepath.endswith(".csv")
    delimiter = "," if is_csv else "\t"
    
    with open(filepath, encoding="utf-8") as f:

        reader = csv.reader(f, delimiter=delimiter)

        column_names = next(reader)

        aliases = {
            "path": ["filename"],
            "accents": ["accent"],
            "text": ["sentence", "raw_transcription", "transcription"],
            "client_id": ["id"],
        }

        for k, v in aliases.items():
            if k not in column_names:
                for alias in v:
                    if alias in column_names:
                        column_names[column_names.index(alias)] = k
                        break

        assert "path" in column_names, f"No path or filename column found in {filepath}."
        assert "text" in column_names, f"No sentence or text column found in {filepath}."
        assert "gender" in column_names, f"No gender column found in {filepath}."
        # assert "client_id" in column_names, f"No client_id column found in {filepath}."
        must_create_client_id = "client_id" not in column_names
        if must_create_client_id:
            column_names.append("client_id")

        path_idx = column_names.index("path")

        checked_files = 0
        for field_values in reader:

            # if data is incomplete, fill with empty values
            if len(field_values) < len(column_names):
                field_values += (len(column_names) - len(column_names)) * [None]

            # set an id if not present
            if must_create_client_id:
                field_values.append(os.path.splitext(field_values[path_idx])[0].replace("/","--"))

            # set absolute path for mp3 audio file
            field_values[path_idx] = os.path.join(path_to_clips, field_values[path_idx])

            if checked_files < max_existence_file_check:
                assert os.path.isfile(field_values[path_idx]), f"Audio file {field_values[path_idx]} does not exist."
                checked_files += 1

            yield {key: value for key, value in zip(column_names, field_values)}


def tsv2kaldi(input_file, audio_folder, output_folder, language=None):
    
    rows = generate_examples(input_file, audio_folder)

    os.makedirs(output_folder, exist_ok=True)

    has_duration = None

    speakers=[]
    with open(output_folder + '/utt2spk', 'w') as utt2spk_file, \
        open(output_folder + '/text', 'w') as text_file, \
        open(output_folder + '/wav.scp', 'w') as wavscp_file, \
        open(output_folder + '/utt2dur', 'w') as utt2dur_file:

        uniq_spks=[]
        for row in rows:
            if has_duration is None:
                has_duration = 'duration' in row
            else:
                assert has_duration == ('duration' in row), "All rows must have the duration or not"

            file_id = os.path.splitext(os.path.basename(row['path']))[0]
            spk_id = row['client_id']
            utt_id = spk_id
            if True: # file_id not in utt_id:
                utt_id += '_'+ file_id
            if spk_id not in uniq_spks:
                uniq_spks.append(spk_id)
                gender = row['gender'][0] if row['gender'] != '' else 'm'
                if row['gender'] == "other":
                    gender = "m"
                if gender not in ["m", "f"]:
                    raise RuntimeError("Unexpected gender: "+row['gender'])
                speakers.append({
                    'id': spk_id,
                    'gender': gender
                })

            text = row["text"]
            if text:
                utt2spk_file.write(utt_id+" "+spk_id+"\n")
                text_file.write(utt_id+" "+text+"\n")
                wavscp_file.write(utt_id+" sox "+ os.path.abspath(row['path']) +" -t wav -r 16k -b 16 -c 1 - |\n")
                if has_duration:
                    utt2dur_file.write(utt_id+" "+row['duration']+"\n")

    if not has_duration:
        os.remove(output_folder + '/utt2dur')

    with open(output_folder + '/spk2gender', 'w') as spk2gender_file:
        for speaker in speakers:
            spk2gender_file.write(speaker['id']+" "+speaker['gender']+"\n")

    return check_kaldi_dir(output_folder, language=language)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Converts a dataset in CSV or TSV format (like CommonVoice) into kaldi format')
    parser.add_argument("input_file", type=str, help="Input TSV or CSV file")
    parser.add_argument("audio_folder", type=str, help="Input folder with audio files inside")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument('--language', default=None, type=str, help='Main language (only for checking the charset and giving warnings)')
    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder    
    audio_folder = args.audio_folder

    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isfile(input_file), f"Input file not found: {input_file}"
    assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"

    tsv2kaldi(input_file, audio_folder, output_folder, language=args.language)
