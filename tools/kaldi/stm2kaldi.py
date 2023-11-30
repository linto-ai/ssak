import os
import random
import pandas as pd
import re 
from linastt.utils.kaldi import check_kaldi_dir


def stm_to_dataframe(stm_file):
    # Read the STM file
    with open(stm_file, "r") as stm_file:
        stm_data = stm_file.read()
    # Split the data into lines
    lines = stm_data.strip().split('\n')

    # Define a pattern to extract information
    pattern = re.compile(r'(\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (.+)')

    # Initialize lists to store data
    segments = []
    for line in lines:
        match = pattern.match(line)
        if match:
            segments.append(match.groups()) 
    # Create a DataFrame
    columns = ['filename', 'channel', 'speaker_id', 'start', 'end', 'gender', 'details', 'transcription']
    stmframe = pd.DataFrame(segments, columns=columns)        
    # Convert numeric columns to float
    numeric_columns = ['start', 'end']
    stmframe[numeric_columns] = stmframe[numeric_columns].astype(float)

    # Replace underscores with hyphens and add '.wav'
    stmframe['filename'] = stmframe['filename'].str.replace('_', '-') + '.wav'
    stmframe['filename']
    # Extract gender information
    stmframe["gender"] = stmframe["gender"].str.extract(r'<\S+,\S+,(.*?)>').replace('female', 'f').replace('male', '')           
               
    return stmframe

def stmframe_to_kaldi(stm_file, audio_path, output_folder):
    stmframe = stm_to_dataframe(stm_file)
    if os.path.isdir(output_folder):
        raise RuntimeError(f"Output folder {output_folder} already exists. Please remove it to overwrite.")
    os.makedirs(output_folder,exist_ok=True)
    utt2spk = []
    spk2utt = []
    text = []
    wav_scp = []
    segments = []
    spk2gender = []
    utt2dur = []
    
    for index, row in stmframe.iterrows():
        utt_id = row['filename'].split('.')[0]
        uniq_utt_id = f'{utt_id}_{index}'
        spk_id = f"{row['speaker_id']}_{index}"
        transcription = row['transcription']
        if transcription is None:
            raise RuntimeError(f"text is non for {uniq_utt_id} utt!!!")
        
        # Assuming 'filename' contains the path to the audio file
        audio_name = os.path.realpath(os.path.join(audio_path, row['filename']))
        assert audio_name,f'This Audio file {row["filename"]} is not exist in {audio_path}'
           
        # Write utt2spk
        utt2spk.append(f'{uniq_utt_id} {spk_id}')
        spk2utt.append(f'{spk_id} {uniq_utt_id}')
        # Write text
        text.append(f'{uniq_utt_id} {transcription}')

        # Write wav.scp
        wav_scp.append(f'{uniq_utt_id} sox {audio_name} -t wav -r 16k -b 16 -c 1 - |')

            # Assuming 'start' and 'end' columns are available
        if not pd.isna(row['start']) and not pd.isna(row['end']):
            start_time = row['start']
            end_time = row['end']
            duration = float(row['end']) - float(row['start'])
            if duration == 0:
                raise RuntimeError(f"Audio file not found for {utt_id} (in {audio_path})")
            # Write segments
            segments.append(f'{uniq_utt_id} {utt_id} {start_time} {end_time}')
            utt2dur.append(f'{uniq_utt_id} {duration}')
            
        # Assuming 'gender' column is available
        if 'gender' in stmframe.columns:
            if row.get("gender") == "other":
                row["gender"] = random.choice(["m", "f"])
            gender = row['gender']
            # Write spk2gender
            spk2gender.append(f'{spk_id} {gender}')
        else:
            gender = random.choice(["m", "f"])
            spk2gender.append(f'{spk_id} {gender}')
        
    # Save the generated data to files
    with open(f'{output_folder}/utt2spk', 'w') as f:
        f.write('\n'.join(utt2spk))

    with open(f'{output_folder}/text', 'w') as f:
        f.write('\n'.join(text))

    with open(f'{output_folder}/wav.scp', 'w') as f:
        f.write('\n'.join(wav_scp))
    if segments:
        with open(f'{output_folder}/segments', 'w') as f:
            f.write('\n'.join(segments))
    if utt2dur:
        with open(f'{output_folder}/utt2dur', 'w') as f:
            f.write('\n'.join(utt2dur))
    if spk2gender:
        with open(f'{output_folder}/spk2gender', 'w') as f:
            f.write('\n'.join(spk2gender))
    
    # check_kaldi_dir(output_folder, language=None)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Converts a dataset in Transcriber format (.xml with extension .trs) into kaldi format')
    parser.add_argument('stmfile', help='Folder with trs files')
    parser.add_argument('audio_folder', help='Folder with audio files (if different from trs folder)', nargs='?', default=None)
    parser.add_argument('output_folder', help='output directory')
    args = parser.parse_args()
    
    assert os.path.isdir(args.audio_folder), f"Folder {args.audio_folder} does not exist"
    assert not os.path.isdir(args.output_folder), f"Folder {args.output_folder} already exists"
    
    stmframe_to_kaldi(args.stmfile, args.audio_folder, args.output_folder)